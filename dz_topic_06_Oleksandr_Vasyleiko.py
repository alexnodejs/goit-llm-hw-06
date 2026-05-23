# %%
"""
# ДЗ Тема 6 — TODO-агент з LangChain v1 + LangGraph `InMemoryStore`

**Мета**: створити TODO-агента з інструментами `add_task`, `list_tasks`,
`delete_task`, які зберігають дані в `InMemoryStore` і доступаються до
нього через `get_store()`. Агент має розуміти україномовні команди й
успішно проходити 5 тестових сценаріїв.

**Середовище** (приклад):
```bash
conda create -n env_mlf python=3.11 spyder spyder-notebook pandas seaborn=0.13 ipywidgets
conda activate env_mlf
pip install langchain langgraph langchain-openai ipynb-py-convert
spyder
```
"""

# %%
# !pip install langchain langgraph langchain-openai

import os
import uuid

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.store.memory import InMemoryStore
from langgraph.config import get_store

os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "your-api-key-here")
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# %%
"""
## Підготовка сховища

`InMemoryStore` — словник у пам'яті. Один інстанс на сесію передається
в `create_agent(store=...)`, тому стан зберігається між викликами агента.
Namespace `("tasks",)` групує всі завдання разом.
"""

# %%
store = InMemoryStore()
NAMESPACE = ("tasks",)

# %%
"""
## Інструмент: додавання завдання
"""

# %%
@tool
def add_task(task: str) -> str:
    """Додає нове завдання до списку. Аргумент `task` — текст завдання."""
    task_id = str(uuid.uuid4())[:8]
    get_store().put(NAMESPACE, task_id, {"task": task})
    return f'Завдання "{task}" було успішно додано. ID: {task_id}'

# %%
"""
## Інструмент: перегляд усіх завдань
"""

# %%
@tool
def list_tasks() -> str:
    """Повертає пронумерований список усіх збережених завдань з їхніми ID."""
    items = get_store().search(NAMESPACE)
    if not items:
        return "У вас немає жодного завдання."
    lines = ["Ось всі ваші завдання:", ""]
    for i, item in enumerate(items, 1):
        lines.append(f"{i}. {item.value['task']} [{item.key}]")
    return "\n".join(lines)

# %%
"""
## Інструмент: видалення завдання за ID
"""

# %%
@tool
def delete_task(task_id: str) -> str:
    """Видаляє завдання за його ID. Якщо ID не знайдено — повертає повідомлення про помилку."""
    item = get_store().get(NAMESPACE, task_id)
    if item is None:
        return f'Завдання з ID "{task_id}" не знайдено.'
    task_text = item.value["task"]
    get_store().delete(NAMESPACE, task_id)
    return f'Завдання "{task_text}" було видалено.'

# %%
"""
## Створення агента

Системний промпт українською пояснює агенту його роль і ключове правило:
для видалення за описом ("видали хліб") треба спершу викликати
`list_tasks`, знайти відповідний ID і лише потім — `delete_task`.
"""

# %%
SYSTEM_PROMPT = """Ти — український TODO-асистент, що керує списком справ користувача.

Ти маєш три інструменти:
- `add_task(task)` — додає нове завдання до списку.
- `list_tasks()` — повертає всі збережені завдання з їхніми ID.
- `delete_task(task_id)` — видаляє завдання за його ID.

Правила:
1. Якщо користувач просить видалити завдання за описом (наприклад, «видали завдання про хліб»), спершу виклич `list_tasks`, знайди серед результатів ID завдання, чий текст відповідає опису, а потім виклич `delete_task` з цим ID.
2. Відповідай користувачу коротко й виключно українською.
3. Коли користувач просить показати список завдань — повертай вивід `list_tasks` без змін, включно з ID у квадратних дужках.
4. Після виконання інших інструментів передавай користувачу зрозумілу відповідь українською, без зайвих технічних деталей.
"""

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENAI_API_KEY"],
    temperature=0,
)

agent = create_agent(
    model=llm,
    tools=[add_task, list_tasks, delete_task],
    system_prompt=SYSTEM_PROMPT,
    store=store,
)

# %%
"""
## Тестові сценарії
"""

# %%
tests = [
    "Додай: купити хліб",
    "Додай: подзвонити лікарю",
    "Покажи всі завдання",
    "Видали завдання про хліб",
    "Що залишилось?",
]

for i, test in enumerate(tests, 1):
    print(f"\n{'='*50}")
    print(f"TEST {i}: {test}")
    print('='*50)
    response = agent.invoke({"messages": [{"role": "user", "content": test}]})
    print(response["messages"][-1].content)

# %%
"""
## Висновки

- Створено TODO-агента, що використовує `InMemoryStore` як довготривалу пам'ять.
- Усі три інструменти (`add_task`, `list_tasks`, `delete_task`) доступаються до сховища через `get_store()` із `langgraph.config` — store автоматично прокидається в контекст графа функцією `create_agent(store=...)`.
- Один інстанс `InMemoryStore` зберігається між викликами `agent.invoke(...)`, тому завдання з TEST 1/2 залишаються доступними у TEST 3, 4, 5.
- Системний промпт навчає агента двокроковому видаленню («listати → видалити за ID»), що дозволяє пройти TEST 4 без ручного передавання ID.
- У продакшні `InMemoryStore` слід замінити на DB-backed store (наприклад, `PostgresStore` або `RedisStore`) для персистентності між рестартами.
"""
