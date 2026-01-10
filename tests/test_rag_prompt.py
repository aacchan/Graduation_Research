# test_rag_prompt.py
import os
import unittest
import importlib

from llm_plan.agent.agent_config import agent_config


class DummyController:
    """LLMを呼ばない（呼んだら落とす）ダミー"""
    async def run(self, *args, **kwargs):
        raise RuntimeError("LLM call is not allowed in this unit test")


def load_agent(substrate_name: str, agent_type: str, llm_type: str, agent_id: str = "player_0"):
    agent_class_path = agent_config[substrate_name][agent_type][llm_type]
    agent_module_path, agent_class_name = agent_class_path.rsplit(".", 1)

    agent_module = importlib.import_module(agent_module_path)
    agent_class = getattr(agent_module, agent_class_name)

    cfg = {"agent_id": agent_id}
    if "hypothetical_minds" in agent_type or "hm" in agent_type:
        cfg["self_improve"] = True

    return agent_class(cfg, DummyController())


def make_minimal_state(agent_id: str = "player_0"):
    # rws_hm_llama3.generate_hls_user_message が参照する最低限の形に合わせる
    opponent_id = "player_1" if agent_id == "player_0" else "player_0"
    ego = {
        f"{agent_id}-north": ((3, 3), "north"),   # v[0] が座標として参照される想定
        f"{opponent_id}-south": ((10, 8), "south"),
        "inventory": [],

        # 観測（例）
        "yellow_box": [(5, 7), (6, 7)],
        "blue_box": [(9, 3)],
        "purple_box": [(2, 12)],
        "ground": [(1, 1), (1, 2)],
    }

    global_state = {
        # movable_locations は wall 以外を集めるので最低限でOK
        "ground": [(x, 0) for x in range(3)],
        "wall": [(99, 99)],
    }

    return {agent_id: ego, "global": global_state}


class TestRAGPrompt(unittest.TestCase):
    def test_hls_prompt_contains_rag_context(self):
        # あなたの main.py の substrate_dict に合わせて rws の正式名を使う
        substrate_name = "running_with_scissors_in_the_matrix__repeated"
        agent_type = os.getenv("AGENT_TYPE", "hm")
        llm_type = os.getenv("LLM_TYPE", "llama3")

        agent = load_agent(substrate_name, agent_type, llm_type, agent_id="player_0")
        state = make_minimal_state("player_0")

        # ---- ここが「RAGが適用できているか」判定の中核 ----
        marker = os.getenv("RAG_MARKER", "RAG_CONTEXT_BEGIN")

        prompt = agent.generate_hls_user_message(state, step=10)

        self.assertIn(marker, prompt, msg=f"RAG marker '{marker}' not found in HLS prompt")

        # 「生の memory dump」を貼っていないこと（ゴミ化しやすいので）
        self.assertNotIn("{'yellow_box'", prompt, msg="Raw dict dump of memory_states is still in prompt")

        # top-k を入れているなら、行数/件数の制限も確認（例：k=20）
        # 例：RAG整形が `- type: ...` の箇条書きなら数を数える
        k = int(os.getenv("RAG_TOPK", "20"))
        num_items = prompt.count("\n- ")
        if num_items > 0:  # 箇条書き形式の場合のみチェック
            self.assertLessEqual(num_items, k, msg=f"RAG context has too many items: {num_items} > {k}")

    def test_feedback_prompt_contains_rag_context(self):
        substrate_name = "running_with_scissors_in_the_matrix__repeated"
        agent_type = os.getenv("AGENT_TYPE", "hm")
        llm_type = os.getenv("LLM_TYPE", "llama3")

        agent = load_agent(substrate_name, agent_type, llm_type, agent_id="player_0")
        state = make_minimal_state("player_0")

        marker = os.getenv("RAG_MARKER", "RAG_CONTEXT_BEGIN")
        prompt = agent.generate_feedback_user_message(state, step=10)

        self.assertIn(marker, prompt, msg=f"RAG marker '{marker}' not found in feedback prompt")
        self.assertNotIn("{'yellow_box'", prompt, msg="Raw dict dump of memory_states is still in feedback prompt")


if __name__ == "__main__":
    unittest.main()
