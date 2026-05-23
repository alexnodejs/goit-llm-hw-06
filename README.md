# GoIT LLM HW-06 — TODO-агент з LangChain v1 + LangGraph InMemoryStore

Домашнє завдання Теми 6 курсу GoIT: агент із трьома інструментами
(`add_task`, `list_tasks`, `delete_task`), що зберігає завдання у
`InMemoryStore` від LangGraph і доступається до сховища через
`get_store()`.

## Файли

- `dz_topic_06_Oleksandr_Vasyleiko.py` — сорс у форматі `ipynb-py-convert` (комірки `# %%`).
- `dz_topic_06_Oleksandr_Vasyleiko.ipynb` — фінальний Jupyter-ноутбук з виконаними виводами.

## Запуск

```bash
conda create -n env_mlf python=3.11 spyder spyder-notebook pandas seaborn=0.13 ipywidgets
conda activate env_mlf
pip install langchain langgraph langchain-openai ipynb-py-convert jupyter

export OPENAI_API_KEY="<your-openrouter-or-openai-key>"
export OPENAI_API_BASE="https://openrouter.ai/api/v1"   # якщо використовуєте OpenRouter

jupyter notebook dz_topic_06_Oleksandr_Vasyleiko.ipynb
```

Якщо хочете перегенерувати `.ipynb` із `.py`:

```bash
ipynb-py-convert dz_topic_06_Oleksandr_Vasyleiko.py dz_topic_06_Oleksandr_Vasyleiko.ipynb
```

## API-ключ

У файлі `.py` ключ читається з `os.environ["OPENAI_API_KEY"]`. Перед
запуском встановіть змінну середовища — у репозиторій ключ не комітиться.

## Тестові сценарії

Агент послідовно виконує 5 запитів:

1. «Додай: купити хліб»
2. «Додай: подзвонити лікарю»
3. «Покажи всі завдання»
4. «Видали завдання про хліб» — агент сам викликає `list_tasks` для пошуку ID, потім `delete_task`.
5. «Що залишилось?»

Виводи збережено у нотбуці.
