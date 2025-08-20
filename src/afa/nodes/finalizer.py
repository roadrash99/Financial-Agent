

import json
from typing import Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage

from afa.config import get_finalizer_llm
from afa.prompts import get_finalizer_prompt, get_system_prompt
from afa.state import ConversationState

__all__ = ["finalizer_node"]


def finalizer_node(state: ConversationState) -> Dict[str, Any]:

    context = {
        "question": state["question"],
        "parsed": state.get("parsed", {}),
        "metrics": state.get("metrics", {})
    }

    messages = [
        SystemMessage(content=get_system_prompt()),
        HumanMessage(content=_build_human_message(context))
    ]

    llm = get_finalizer_llm()
    response = llm.invoke(messages)

    final_answer = response.content.strip()
    
    return {"final_answer": final_answer}


def _build_human_message(context: Dict[str, Any]) -> str:
    prompt = get_finalizer_prompt()
    context_json = json.dumps(context, indent=2, sort_keys=True)
    
    return f"{prompt}\n\nJSON: {context_json}"
