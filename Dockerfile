FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock* ./

RUN uv pip install --system --no-cache -r pyproject.toml

COPY . .

CMD ["python", "-m", "src.example_llm_rag"]
