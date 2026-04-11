# Plan & Architect Agent

Your **AI backend architect assistant**. It talks to you about what you're building, then generates a complete project plan — right in your terminal. Runs 100% locally. No API keys. No internet needed after setup.

---

## Quick Start (TL;DR)

> **Already have Python 3.10+ and Ollama? Run these 5 commands:**
>
> ```bash
> ollama pull llama3.1
> unzip planagent.zip        # or: git clone <repo-url>
> cd planagent
> python3 -m venv venv && source venv/bin/activate && pip install -e .
> planagent /path/to/your/project
> ```

If that doesn't make sense yet, follow the full guide below.

---

## How It Works (Simple Version)

```
You install this agent once on your PC
            ↓
You open terminal in VS Code (or any terminal)
            ↓
You run: planagent /path/to/your/project
            ↓
Agent scans your project and asks you questions:
  - What are you building?
  - Who are the users?
  - What features do you need?
  - What tech stack?
            ↓
Agent generates a full architecture plan
            ↓
Files appear inside your project:
  your-project/.planagent/
    ├── plan.md            (what to build)
    ├── architecture.md    (how to build it)
    ├── api_contracts.md   (API endpoints)
    ├── roadmap.md         (v1/v2 checklist)
    └── context.json       (machine-readable)
```

**You don't open the agent in VS Code. You just run it from the terminal.**

---

## Full Setup Guide (First Time Only)

### Step 1: Install Python

Check if you already have it:

```bash
python3 --version
```

Need **3.10 or higher**. If not installed: [https://python.org/downloads](https://python.org/downloads)

### Step 2: Install Ollama

Ollama runs the AI model locally on your computer.

- Download from [https://ollama.com](https://ollama.com)
- Install it like a normal app
- Then open terminal and download the AI model (one-time, ~4.7 GB):

```bash
ollama pull llama3.1
```

### Step 3: Get the Agent

**Option A — From a ZIP file:**

```bash
unzip planagent.zip
cd planagent
```

**Option B — From GitHub:**

```bash
git clone https://github.com/<username>/planagent.git
cd planagent
```

### Step 4: Install the Agent

```bash
python3 -m venv venv

# Mac/Linux:
source venv/bin/activate

# Windows:
# venv\Scripts\activate

pip install -e .
```

### Step 5: Check It Works

```bash
planagent --help
```

If you see this, you're all set:

```
Usage: planagent [OPTIONS] [PATH]
  Plan & Architect Agent — designs your backend through conversation.
```

---

## Using the Agent (Every Time)

### 1. Make sure Ollama is running

Ollama usually runs in the background after installation. If not:

```bash
ollama serve
```

### 2. Activate the virtual environment

```bash
source ~/planagent/venv/bin/activate
```

(Replace `~/planagent` with wherever you unzipped/cloned it)

### 3. Run the agent on your project

```bash
planagent /path/to/your/project
```

Or if you're already inside your project folder:

```bash
planagent .
```

### 4. Answer the questions

The agent will ask you 2-3 questions at a time about your project. Just answer naturally. When it has enough info, it generates the plan automatically.

---

## Example Session

```
 Plan & Architect Agent v1.0
--------------------------------------------------
Scanning /home/user/my-food-app...
No existing files found. Starting Fresh.
--------------------------------------------------

Agent: What are you building? Even one line is fine to start.

You: A food delivery app with customer and driver roles

Agent: Got it! A food delivery platform. Let me ask a few things:
1. Will restaurants manage their own menus, or will you handle that?
2. Do you need real-time order tracking for drivers?
3. Any preference on tech stack — Python/Node/Go?

You: Yes restaurants manage menus, yes real-time tracking, use Python + FastAPI

Agent: ...keeps asking until it has everything...

Agent: CONVERSATION_COMPLETE
       Generating plan...

  ✅ Wrote plan.md
  ✅ Wrote architecture.md
  ✅ Wrote api_contracts.md
  ✅ Wrote roadmap.md
  ✅ Wrote context.json

  All done.
```

After this, open your project in VS Code — you'll see a `.planagent/` folder with all the docs.

---

## Where to Put This Agent on Your PC

**Do NOT put it inside your project.** Keep them separate:

```
/home/you/
├── planagent/              ← agent lives here (installed once)
├── my-food-app/            ← your project
├── my-other-project/       ← another project
```

The agent creates its output **inside** whichever project you point it at.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `planagent: command not found` | Run `source ~/planagent/venv/bin/activate` first |
| `ModuleNotFoundError: No module named 'typer'` | Run `pip install -e .` inside the planagent folder |
| `Connection refused` or LLM errors | Start Ollama: `ollama serve` |
| Model not found | Download it: `ollama pull llama3.1` |

---

## Knowledge Base

The agent ships with a **pre-built system design knowledge base** (architecture patterns, scaling strategies, design best practices). No setup needed — it's always available during conversations.

To regenerate the pre-built index after editing source files:

```bash
python -m planagent.knowledge.prebuild
```

---

## Tech Stack

- **AI Model**: Llama 3.1 via Ollama (runs locally, no API keys)
- **LLM Router**: LiteLLM
- **CLI**: Typer
- **Terminal UI**: Rich
- **Language**: Python

---

## Project Structure

```
planagent/
├── planagent/
│   ├── cli.py                  # CLI entry point
│   ├── context_reader.py       # Scans project folder
│   ├── conversation_manager.py # Conversation loop with LLM
│   ├── plan_generator.py       # Generates architecture plan
│   ├── output_writer.py        # Writes markdown & JSON output
│   ├── llm.py                  # LLM wrapper + token tracking
│   ├── state.py                # State dictionary definition
│   └── prompts/
│       ├── system_prompt.txt   # Conversation system prompt
│       └── plan_template.txt   # Plan generation template
├── tests/
├── pyproject.toml
├── requirements.txt
└── README.md
```
