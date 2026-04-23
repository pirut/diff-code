from __future__ import annotations

import torch


def generate(
    model,
    initial_tokens: torch.LongTensor,
    *,
    attention_mask: torch.Tensor | None = None,
    initial_mask: torch.BoolTensor | None = None,
    steps: int = 8,
    temperature: float = 0.8,
    top_k: int | None = 50,
    top_p: float | None = 0.95,
    confidence_threshold: float | None = None,
    remask_fraction: float = 0.5,
    mask_token_id: int,
) -> torch.LongTensor:
    model.eval()
    tokens = initial_tokens.clone()
    original_mask = initial_mask if initial_mask is not None else tokens.eq(mask_token_id)
    active_mask = original_mask.clone()

    if attention_mask is None:
        attention_mask = torch.ones_like(tokens, dtype=torch.long)

    with torch.no_grad():
        for step in range(steps):
            if not active_mask.any():
                break

            outputs = model(
                tokens,
                attention_mask=attention_mask,
                mask_positions=active_mask,
                use_cache=False,
            )
            logits = outputs.logits
            sampled_ids, confidence = _sample_tokens(
                logits=logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            tokens[active_mask] = sampled_ids[active_mask]

            if step == steps - 1:
                break

            next_mask = torch.zeros_like(active_mask)
            candidate_confidence = confidence.masked_fill(~original_mask, 1.0)

            if confidence_threshold is not None:
                next_mask = original_mask & (candidate_confidence < confidence_threshold)
            else:
                total_original = int(original_mask.sum().item())
                remaining_fraction = ((steps - step - 1) / max(steps - 1, 1)) * remask_fraction
                remask_count = int(total_original * remaining_fraction)
                if remask_count > 0:
                    flat_confidence = candidate_confidence.view(-1)
                    flat_mask = original_mask.view(-1)
                    original_indices = torch.nonzero(flat_mask, as_tuple=False).flatten()
                    if original_indices.numel() > 0:
                        ranked = original_indices[torch.argsort(flat_confidence[original_indices])]
                        chosen = ranked[:remask_count]
                        next_mask.view(-1)[chosen] = True

            if next_mask.any():
                tokens[next_mask] = mask_token_id
            active_mask = next_mask

    return tokens


def _sample_tokens(
    *,
    logits: torch.Tensor,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
) -> tuple[torch.LongTensor, torch.Tensor]:
    if temperature <= 0:
        probabilities = torch.softmax(logits, dim=-1)
        token_ids = probabilities.argmax(dim=-1)
        confidence = probabilities.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
        return token_ids, confidence

    scaled_logits = logits / max(temperature, 1e-5)
    filtered_logits = _apply_sampling_filters(scaled_logits, top_k=top_k, top_p=top_p)
    probabilities = torch.softmax(filtered_logits, dim=-1)

    flat_probs = probabilities.view(-1, probabilities.size(-1))
    sampled = torch.multinomial(flat_probs, num_samples=1).view(probabilities.shape[:-1])
    confidence = probabilities.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
    return sampled, confidence


def _apply_sampling_filters(
    logits: torch.Tensor,
    *,
    top_k: int | None,
    top_p: float | None,
) -> torch.Tensor:
    filtered = logits.clone()

    if top_k is not None and top_k > 0 and top_k < filtered.size(-1):
        top_values, _ = torch.topk(filtered, top_k, dim=-1)
        kth_values = top_values[..., -1, None]
        filtered = filtered.masked_fill(filtered < kth_values, float("-inf"))

    if top_p is not None and 0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        remove_mask = cumulative_probs > top_p
        remove_mask[..., 1:] = remove_mask[..., :-1].clone()
        remove_mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))
        filtered.scatter_(-1, sorted_indices, sorted_logits)

    return filtered
