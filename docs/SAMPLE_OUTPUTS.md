# Sample outputs

Raw, unedited generations from `out/qwen7b-forge-mixed.gguf`, with **FP16 generations from
the same prompts as a control**. Committed so the quality of a FORGE checkpoint is
auditable without re-running the evaluation harness.

## How these were produced

```bash
llama-simple -m out/qwen7b-forge-mixed.gguf -n 100 "<prompt>"
```

`llama-simple` does **raw completion with greedy decoding**. That matters twice over:

* Qwen2.5-Coder-7B is a **base** model. `llama-cli` and `llama-completion` apply a chat
  template by default, which makes a base model emit degenerate output *regardless of
  quantization*. Always use a raw-completion path for base weights.
* Greedy decoding has no repetition penalty, which makes loops more likely. Sampling
  (`--temp 0.7 --top-p 0.95 --repeat-penalty 1.1`, seed 42) was also tried and did not
  rescue the FORGE outputs, while FP16 remained clean through the identical path.

## Headline: perplexity does not predict this

> **`ffn_down`-mixed FORGE reaches 9.4150 ppl (1.50x FP16) yet produces zero
> syntactically valid Python across 8 standard prompts.**

| | parses as Python | degenerate repetition |
|---|---:|---:|
| FP16 | **6 / 8** | 0 / 8 |
| **FORGE mixed** | **0 / 8** | **3 / 8** |

FP16's two non-parsing samples are simply truncated at the 100-token budget, not broken.
FORGE's failures are structural: dropped indentation, unclosed parentheses, and repetition
loops.

This is the risk the project plan called out from the start -- *"code generation is less
forgiving than perplexity; report both, never claim quality from PPL alone."* It is now
measured rather than hypothetical, and it is the single most important number in this
repository.

## Side by side

### Prompt 1

```python
def fibonacci(n):
```

**FP16** (control)

```python
def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

def lucas(n):
    if n == 0:
        return 2
    elif n == 1:
        return 1
    else:
```

**FORGE mixed**

```python
def fibonacci(n):
    if n == 0:
    return 0
    if n == 1:
    return 1
    return fibonacci(n-1 + fibonacci(n-2

def fibonacci(n):
    if n == 0:
    return 0
    if n == 1:
    return 1
    return fibonacci(n-1 + fibonacci n-2
```

---

### Prompt 2

```python
def binary_search(arr, target):
```

**FP16** (control)

```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
```

**FORGE mixed**

```python
def binary_search(arr, target):
    def binary_search(arr, target):
    def binary_search(arr, target):

    def binary search(arr, target)
    def binary search(arr, target)
    def binary search(arr, target)
    def binary search(arr, target)
    def binary search(arr, target)
    def binary search(arr, target)
    def binary search(arr, target)
    def binary search(arr, target)
    def binary search(arr, target)
```

---

### Prompt 3

```python
def quicksort(arr):
```

**FP16** (control)

```python
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

# Example usage:
arr = [3, 6, 8, 10, 
```

**FORGE mixed**

```python
def quicksort(arr):
    if len(arr) <= 1:
        return arr

    else:
        pivot = arr[0]
    left = []

    right = []

    for i in arr:
        if i < pivot:
        left.append(i)
    else:
```

---

### Prompt 4

```python
def is_palindrome(s):
```

**FP16** (control)

```python
def is_palindrome(s):
    return s == s[::-1]

def longest_palindrome(s):
    n = len(s)
    if n == 0:
        return ""
    if n == 1:
        return s
    max_len = 1
    start = 0
    for i in range(n):
        for j in range(i, n):
            if is_palindrome(s[i:j+1]) and j-i+1 > max_len:
```

**FORGE mixed**

```python
def is_palindrome(s):
    return s == s[::-1

def is_palindrome(s):
    return s == s[::-1

def is_palindrome(s):
    return s == s[::-1

def is_palindrome(s):
    return s == s[::-1

def is_palindrome(s):
    return s == s[::-1
```

---

### Prompt 5

```python
def gcd(a, b):
```

**FP16** (control)

```python
def gcd(a, b):
    if b == 0:
        return a
    else:
        return gcd(b, a % b)


def lcm(a, b):
    return a * b // gcd(a, b)


def solution(arr):
    answer = arr[0]
    for i in range(1, len(arr)):
```

**FORGE mixed**

```python
def gcd(a, b):
    if b == 0:
    return a
    else:
    return gcd(a, b)
    if b == 0:
    return a
    else:
    return gcd(a, b)
    if b == 0:
    return a
    else:
    return gcd(a, b)
    if b == 0:
```

---

### Prompt 6

```python
class Stack:
```

**FP16** (control)

```python
class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        return self.items[-1]

    def is_empty(self):
```

**FORGE mixed**

```python
class Stack:
    def __init__(self):

    def __init__(self):

    def __init__(self):
    def __init__(self):
    def __init__(self):
    def __init__(self):
    def __init__(self):
    def __init__(self):
    def __init__(self):
    def __init__(self):
    def __init__(self):
```

---

### Prompt 7

```python
def merge_sort(arr):
```

**FP16** (control)

```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left_half = arr[:mid]
    right_half = arr[mid:]

    left_half = merge_sort(left_half)
    right_half = merge_sort(right_half)

    return merge(left_half, right_half)

def merge(left, right):
```

**FORGE mixed**

```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    else:
        mid = len(arr)/2
        left = merge_sort(arr[:mid]
        right = merge_sort(arr[mid]
        return merge(left, right)

def merge(arr1, arr2):

    if arr1 is empty then return arr2
    else if arr2 is empty then return arr1
```

---

### Prompt 8

```python
def factorial(n):
```

**FP16** (control)

```python
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)

def sum_of_digits(n):
    if n < 10:
        return n
    else:
        return n % 10 + sum_of_digits(n // 10)

number = 100
factorial_result = factorial(number)
```

**FORGE mixed**

```python
def factorial(n):
    if n == 0:
    return 1
 else:
    return n * factorial(n - 1

def factorial(n):
    if n == 0:
    return 1
 else:
    return n * factorial(n - 1

def factorial(n):
    if n == 0:
```

---

## Reproducing the audit

```bash
# build the reference F16 GGUF
python llamacpp/llama.cpp/convert_hf_to_gguf.py <qwen7b-snapshot> \
    --outfile out/qwen7b-f16.gguf --outtype f16

# generate both sides
for p in "def fibonacci(n):" "def binary_search(arr, target):" ...; do
    llama-simple -m out/qwen7b-f16.gguf          -n 100 "$p"
    llama-simple -m out/qwen7b-forge-mixed.gguf  -n 100 "$p"
done
```

Syntactic validity is checked by taking each generation up to the start of a second
top-level `def`/`class` and running it through `ast.parse`.
