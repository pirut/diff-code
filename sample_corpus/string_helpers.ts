export function greet(name: string): string {
  return `hello ${name}`;
}

export function clamp(value: number, min: number, max: number): number {
  if (value < min) {
    return min;
  }
  if (value > max) {
    return max;
  }
  return value;
}

export function sum(items: number[]): number {
  let total = 0;
  for (const item of items) {
    total += item;
  }
  return total;
}
