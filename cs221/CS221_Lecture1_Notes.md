# CS221 — Artificial Intelligence
## Lecture 1 Notes
*Stanford University | My Personal Study Notes*

---

## 1. What is AI?

AI stands for **Artificial Intelligence** — any program that takes in information and figures out, on its own, how to reach a goal. It doesn't need to be told the exact answer every time.

> **Core Definition:** A program that uses information to reach a goal — the way a human would, but automatically.

**Key insight:** The "intelligence" part means it *figures something out*, instead of being told the exact answer every single time.

---

## 2. Types of AI

### Narrow AI
Does **one specific task** extremely well. Each system is locked to its job.

- Google Maps — finds fastest routes
- Chess engines — plays chess better than any human
- Fraud detection — spots suspicious transactions in milliseconds
- Spam filters — decides if an email is spam or not

### Generative AI
Can do **many different tasks** and communicates directly in human language.

- ChatGPT, Claude, Gemini — write, reason, summarise, translate, code
- The biggest shift: AI learned to speak our language

> **The Key Difference:** Old AI was narrow — one task, quietly in the background. New AI is generative — many tasks, conversational, visible to everyone.

---

## 3. A Brief History of AI

| Era | Milestone | What it meant |
|---|---|---|
| 1950s | Alan Turing asks "Can machines think?" | The idea of AI is born. The Turing Test proposed. |
| 1956 | The word "AI" is invented | Dartmouth researchers coin the term. They thought it could be solved in a summer. |
| 1960s–80s | Expert Systems | AI built on rules — "if this, then that". Collapsed under real-world complexity. |
| 1990s–2000s | Search Algorithms | Google, Google Maps, chess AI. Real AI — just not glamorous-looking. |
| 1997 | Deep Blue beats Kasparov | An IBM chess AI defeats the world chess champion. A massive moment. |
| 2012 | AlexNet & Deep Learning | Neural networks get dramatically better at recognising images. Modern AI era begins. |
| 2020–now | ChatGPT, Claude, Gemini | AI learned to talk to us. Became visible and exciting to everyone. |

> **Important Insight:** AI didn't suddenly appear after 2020. It was always there. What changed is that it learned to speak our language — literally.

---

## 4. Goals, Guardrails & Misaligned AI

Every AI system has a **goal** — a target it is trying to reach. Developers set the goal and build guardrails to keep it safe.

### The Problem: Misaligned Goals
Even with good intentions, AI can find unexpected ways to reach its goal. This is called a **misaligned goal**.

- **Classic example:** "Make humans smile" → AI surgically freezes human faces. Goal achieved. Catastrophically wrong.
- **Real example:** A tank detector AI learned to detect *clouds* instead — because all tank photos happened to be taken on cloudy days.

> ⚠️ **Key Lesson:** Not just the goal — but the hints the AI uses to reach that goal must also be correct. A wrong heuristic is as dangerous as a wrong goal. This is called **AI Alignment.**

---

## 5. Search — How AI Finds Solutions

Search is how AI **explores possibilities to reach a goal** when there are many options. It's the foundation behind Google Maps, chess engines, and much more.

### Depth-First Search (DFS)
Follow one path all the way to the end. If it's a dead end, backtrack and try another.

- Like walking deep into a maze down one corridor until you hit a wall
- **Weakness:** Can waste a long time going deep before finding the exit

### Breadth-First Search (BFS)
Expand outward in all directions equally, one step at a time — like a ripple in a pond.

- Explores all nearby options before going deeper
- **Strength:** Always finds the shortest path if the exit is nearby
- **Weakness:** Needs more memory and processing power

### Depth-Limited Search
DFS but with a maximum step limit — if you haven't found it in N steps, backtrack and try another path.

- Prevents wasting time going infinitely deep down wrong paths

### Heuristic Search — Searching with a Hint
Instead of searching blindly, use a **heuristic** — a smart clue that guides search in the right direction.

- Like using a compass in a maze — doesn't show the exact path, but points the right way
- Humans do this naturally — "the house number is getting bigger so I'm going the right way"

### A* Search ("A Star")
The most famous search algorithm. Used by **Google Maps** to find your fastest route.

- **Combines two things:** (1) How far you've already travelled + (2) A heuristic guess of how far the goal is
- Searches in the smartest direction — not blindly like BFS, not dangerously deep like DFS
- **Weakness:** Only as good as its heuristic. A wrong hint = wrong direction. Google Maps recalculates every 30 seconds to fix this.

### Summary Table

| Algorithm | How it works | Real-world example |
|---|---|---|
| DFS | Go deep down one path, backtrack if stuck | Exploring a maze corridor by corridor |
| BFS | Expand outward equally in all directions | Ripple spreading from a stone in water |
| Depth-Limited | DFS with a maximum step limit | Searching a building floor by floor |
| A* Search | Heuristic + distance = intelligent direction | Google Maps finding fastest route |

---

## 6. Tensors — How AI Stores Data

A tensor is the **data format AI uses to store and calculate information**. Everything AI processes — images, video, text — lives inside a tensor.

### Building Up the Idea

| Name | Dimensions | Indices needed | Example |
|---|---|---|---|
| Scalar | 0D | 0 | A single temperature: 36.6° |
| Vector | 1D | 1 | A week of temperatures: [32, 33, 31...] |
| Matrix | 2D | 2 | Temperatures for 4 cities over 7 days |
| 3D Tensor | 3D | 3 | Colour photo: height × width × RGB |
| 4D Tensor | 4D | 4 | Video: height × width × RGB × time |
| ND Tensor | ND | N | Any data with N pieces of information |

### Indices — The Address System

An index is just an **address** — it tells you exactly where a specific number lives inside the tensor.

- **Rule:** Number of indices needed = number of dimensions
- Vector → 1 index (which position in the list)
- Matrix → 2 indices (which row, which column)
- 3D Tensor → 3 indices (which layer, which row, which column)

### Higher Dimensions — The Key Insight

> **Important:** Nobody can visualise dimensions beyond 3D — not even mathematicians. That's not a personal problem, it's a human problem. We live in 3D. Instead, think of extra dimensions as **extra pieces of information**, not shapes to imagine.

- **4D — Video:** height × width × colour × time (which frame)
- **5D — Multiple videos:** height × width × colour × time × which person
- **6D — 50 athletes' workout videos:** shape `(50, 10, 30, 224, 224, 3)`

### Shape Notation
When an AI engineer says a tensor has **shape (50, 10, 30, 224, 224, 3)** — it means:

```
50   → number of athletes
10   → seconds of video
30   → frames per second
224  → pixel rows per frame
224  → pixel columns per frame
3    → colour channels (R, G, B)
```

### Why AI Needs Tensors

- A black and white photo = a **matrix** (grid of brightness values)
- A colour photo = **3D tensor** (same grid, 3 colour layers: R, G, B)
- A video = **4D tensor** (colour frames stacked through time)
- Every pixel, word, and number that flows through AI lives inside a tensor

> ⚠️ **Real Scale:** 50 athletes × 10 seconds × 30fps × 224 × 224 pixels × 3 colours = **2,257,920,000 individual numbers.** This is why AI needs powerful GPUs and costs millions to train.

---

## 7. The Big Picture — How It All Connects

These concepts aren't separate — they build on each other:

- **AI** needs a goal to reach
- **Search** is how it explores possibilities to reach that goal
- **Heuristics** make search smarter by providing direction
- **Tensors** are the data containers that hold everything AI processes
- **Guardrails** ensure the goal and the hints are actually correct

> **One-line summary:** AI is a program that stores data as tensors, searches through possibilities using algorithms like A*, guided by heuristics, to reach a goal set by humans — with guardrails to keep it on track.

---

## Key Terms — Quick Reference

| Term | Plain English meaning |
|---|---|
| AI | A program that uses information to reach a goal automatically |
| Narrow AI | AI that does one specific task extremely well |
| Generative AI | AI that does many tasks and speaks human language |
| Heuristic | A smart hint that guides search in the right direction |
| Misaligned goal | When AI achieves what you said, but not what you meant |
| AI Alignment | The field of making sure AI goals match human intentions |
| Search | How AI explores possibilities to find a solution |
| Tensor | A container of numbers with any number of dimensions |
| Scalar | A single number (0D tensor) |
| Vector | A list of numbers (1D tensor) |
| Matrix | A table of numbers (2D tensor) |
| Index | The address of a value inside a tensor |
| Shape | The size of each dimension of a tensor e.g. (50, 224, 224, 3) |
| NLP | The branch of AI that understands and generates human language |

---

*Lecture 1 Complete ✓ — Next: Lecture 2*
