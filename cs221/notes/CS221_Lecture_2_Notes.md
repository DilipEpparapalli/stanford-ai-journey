
### CS221 — Tensors, Einsum, Gradients & Backpropagation
---

## 1. Tensors — The Atoms of AI

> **Everything in AI is stored as a tensor** — data, parameters, mistakes, everything.

A tensor is simply a **box that holds numbers in an organized shape.**

### Orders of Tensors

|Order|Name|What it looks like|Example|
|---|---|---|---|
|0|Scalar|A single number|`42`|
|1|Vector|A list of numbers|`[1, 2, 3]`|
|2|Matrix|A grid of numbers|A spreadsheet|
|3+|Tensor|A cube or higher|Multiple spreadsheets stacked|

### What is an Axis?

An **axis** is a direction you can move through the tensor.

- A spreadsheet has 2 axes:
    - **Axis 0** = up/down (rows)
    - **Axis 1** = left/right (columns)

### In Your Own Words

> _"A tensor is the data format AI uses for everything. Order means how many dimensions the tensor has — 0 is a single number, 1 is a list, 2 is a grid, 3+ is a cube. Axis is the direction you can move through the tensor."_

---

## 2. Einsum — The One Master Operation

> **Einsum is one command that replaces many operations.** Instead of remembering separate functions for dot products, matrix multiplication, etc. — einsum does all of them.

### The One Rule That Explains Everything

> For every combination of input positions, **multiply** the numbers at those positions, then **dump the result** into the matching output position.
> 
> - If an axis **appears in the output** → keep it
> - If an axis **does NOT appear in the output** → add everything up and collapse it

### Einsum String Format

```
"input_axes -> output_axes"
```

- Left of `->` = input axes
- Right of `->` = output axes
- Comma separates multiple input tensors

---

### All Einsum Operations Explained

#### 1. Identity — "Just copy it"

```
"i -> i"
```

Input list goes out unchanged. For each position i, copy the number to the same position in output.

---

#### 2. Sum — "Collapse everything into one number"

```
"i ->"
```

Right side is **empty** → output is a single number. All elements pile into one scalar.

---

#### 3. Element-wise Product — "Multiply position by position"

```
"i, i -> i"
```

Two lists. For each position i, multiply matching numbers. Result stays at position i.

`[0, 1, 10] × [0, 1, 10] = [0, 1, 100]`

---

#### 4. Dot Product — "Multiply then collapse"

```
"i, i ->"
```

Same as element-wise product but output is empty → everything collapses to one number.

`0×0 + 1×1 + 10×10 = 101`

---

#### 5. Outer Product — "Every combination of two lists"

```
"i, j -> i j"
```

Try **every pair** of positions (i from list 1, j from list 2). Result is a grid (matrix). Like a multiplication table.

---

#### 6. Matrix × Vector

```
"i j, j -> i"
```

For each row i of the matrix, multiply each element by the matching vector element, add them all up. The `j` disappears → gets summed away.

---

#### 7. Matrix × Matrix

```
"i k, j k -> i j"
```

For each pair of rows, multiply matching elements across shared axis k and sum. The `k` disappears → summed away.

---

### The Pattern To Remember

|What disappears on the right|What happens|
|---|---|
|Nothing disappears|Rearranging or copying|
|One axis disappears|Summing along that direction|
|Everything disappears|End up with one number|

### In Your Own Words

> _"Einsum is one command that replaces many operations. You give it a string that maps input axes to output axes. For every combination of positions, it multiplies the numbers and puts the result in the matching output position. If an axis is missing from the output side, it adds everything up and collapses that dimension."_

---

## 3. Gradients — Which Way Is Uphill?

### The Core Idea

> The gradient answers one question: **"If I change this input a tiny bit, how much does the output change — and in which direction?"**

### The Mountain Analogy

Imagine you're lost in the mountains at night trying to reach the valley (lowest point):

- You feel the ground — find which direction slopes downward
- Take a small step that way
- Repeat

**That is gradient descent.** The gradient tells you which direction is uphill.

### Loss Function

In AI, instead of mountains you have a **loss function.**

- **High loss** = AI is very wrong
- **Low loss** = AI is doing well
- **Goal** = find the settings (weights) that make loss as small as possible

### Partial Derivative

When you have multiple inputs (like weight1 and weight2):

> **Partial derivative** = change ONE input at a time and watch what happens to the output.

The **gradient** = ALL partial derivatives bundled together. Same shape as the input.

### The Key Formula

For the function **y = x²**, the derivative (slope) is **2x**.

This means: at any point, if x changes by a tiny bit, y changes by **2x times** that amount.

|Value of x|Blame (gradient)|
|---|---|
|0|0 (perfectly flat, at the bottom)|
|1|2|
|4|8|
|5|10|

> **The closer you are to the bottom, the smaller the gradient.** The AI naturally slows down as it approaches the answer.

---

## 4. Computation Graphs

### Why We Need Them

Complex AI functions are built from simple operations (add, multiply, square, etc.). Instead of manually calculating gradients for the whole thing, we:

1. Break the function into a graph of small steps
2. Let an algorithm handle all the gradient calculations automatically

This is called **backpropagation** — and it's what libraries like PyTorch do under the hood.

---

## 5. Forward Pass & Backward Pass

### Forward Pass — "Calculate the answer"

Go from **inputs → output**, step by step. Each node calculates its value based on what came before it.

**Example:** `f(x1, x2) = (x1 + x2)²` with x1=2, x2=3

```
x1 = 2
x2 = 3
sum = x1 + x2 = 5      ← calculate this
y = sum² = 25           ← calculate this
```

---

### Backward Pass — "Assign blame"

Go from **output → inputs**, calculating how much each node contributed to the final result.

```
y    → blame = 1   (it IS the output, fully responsible for itself)
sum  → blame = 10  (2 × current value of sum = 2 × 5 = 10)
x1   → blame = 10  (chain rule: 1 × 10 = 10)
x2   → blame = 10  (chain rule: 1 × 10 = 10)
```

> **Blame = gradient** = if this node's value changes by 1, the output changes by this much.

---

### The Chain Rule — How Blame Flows Backwards

> If A affects B, and B affects C — then to find how much A affects C: **Multiply** (how much A affects B) × (how much B affects C)

That's it. Backpropagation just applies this rule automatically across every node.

---

## 6. Backpropagation — The Full Algorithm

```
Step 1: Build a computation graph
        x1 → add → square → y (the loss)
        x2 ↗

Step 2: Forward Pass
        Calculate every value left to right
        x1=2, x2=3, sum=5, y=25

Step 3: Backward Pass
        Assign blame right to left using chain rule
        y=1, sum=10, x1=10, x2=10

Step 4: Gradient Descent
        Nudge weights in the OPPOSITE direction of blame
        new weight = old weight - (learning rate × blame)

Step 5: Repeat thousands of times
        Each time loss gets smaller → AI gets better
```

---

## 7. Gradient Descent

### The Goal

Find the weights that make the loss as small as possible.

### The Update Rule

```
new weight = old weight - (learning rate × gradient)
```

- **Subtract** because you want to go downhill (opposite of gradient)
- **Learning rate** controls how big each step is

### The Learning Rate Trade-off

```
Too small  → takes forever, but gets there safely
Too large  → overshoots, bounces around, never arrives (diverges!)
Just right → reaches the bottom efficiently ✅
```

**The overshooting problem visualised:**

```
\         /
 \       /
  ←-------→   ← jump OVER the bottom
  ←-----------→  ← now even higher on the other side!
```

With too large a learning rate, the loss doesn't go down — it **bounces or explodes.**

---

## 8. The Full Story — Cooking Analogy

|AI Concept|Cooking Analogy|
|---|---|
|Loss|How bad the dish tastes|
|Forward pass|Cook it and taste it|
|Backward pass|Think back — was it the salt? The heat?|
|Gradient|How much each ingredient caused the bad taste|
|Gradient descent|Adjust each ingredient a little next time|
|Learning rate|How boldly you adjust — too much ruins it differently|

> Repeat enough times and you become a great cook. That's machine learning.

---

## 9. Summary

|Concept|Plain English|
|---|---|
|Tensor|The data format AI uses for everything|
|Order|How many dimensions the tensor has|
|Axis|A direction you can move through the tensor|
|Einsum|One master operation replacing many|
|Gradient|Which direction is uphill, and by how much|
|Partial derivative|How much ONE input affects the output|
|Forward pass|Calculate values from input to output|
|Backward pass|Assign blame from output back to inputs|
|Chain rule|Multiply how A affects B by how B affects C|
|Backpropagation|Algorithm that automates the chain rule across a graph|
|Loss function|How wrong the AI is right now|
|Gradient descent|Nudge weights downhill to reduce loss|
|Learning rate|Size of each step downhill|
