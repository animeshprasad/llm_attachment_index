# LLM Attachment Index

A Python project for LLM attachment indexing.

## Features
- OpenAI integration
- Anthropic Claude support
- Transformer-based processing
- Web search capabilities via DuckDuckGo

## Setup and Installation

This project uses Poetry for dependency management. To get started:

1. Install Poetry:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Clone the repository:
```bash
git clone https://github.com/animeshprasad/llm_attachment_index.git
cd llm_attachment_index
```

3. Install dependencies:
```bash
poetry install
```

4. Activate the virtual environment:
```bash
poetry shell
```

## Project Structure

```
llm_attachment_index/
├── src/
│   └── llm_attachment_index/
│       ├── __init__.py
│       └── main.py
├── tests/
│   ├── __init__.py
│   └── test_main.py
├── pyproject.toml
├── .gitignore
└── README.md
```

## Development

- Format code:
```bash
poetry run black .
```

- Run linter:
```bash
poetry run flake8
```

- Run tests:
```bash
poetry run pytest
```

## License

MIT License