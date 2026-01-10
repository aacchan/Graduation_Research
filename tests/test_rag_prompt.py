# tests/test_rag_prompt.py
import os
import unittest
import importlib

from llm_plan.agent.agent_config import agent_config


class DummyController:
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
    opponent_id = "player_1" if agent_id == "player_0" else "player_0"

    ego = {
        f"{agent_id}-north": ((3, 3), "north"),
        f"{opponent_id}-south": ((10, 8), "south"),
        "inventory": [],

        "yellow_box": [(5, 7), (6, 7)],
        "blue_box": [(9, 3)],
        "purple_box": [(2, 12)],
        "ground": [(1, 1), (1, 2)],
    }

    global_state = {
        "ground": [(0, 0), (1, 0), (2, 0)],
        "wall": [(99, 99)],
    }

    return {agent_id: ego, "global": global_state}


class TestRAGPrompt(unittest.TestCase):
    def setUp(self):
        self.substrate_name = "running_with_scissors_in_the_matrix__repeated"
        self.agent_type = os.getenv("AGENT_TYPE", "hm")
        self.llm_type = os.getenv("LLM_TYPE", "llama3")
        self.marker = os.getenv("RAG_MARKER", "RAG_CONTEXT_BEGIN")  # RAG導入後にpromptに入れる文字列

    def _prime_memory(self, agent, state, step):
        # 実環境ではstepごとに呼ばれることが多いので、テストでも明示的に呼ぶ
        if hasattr(agent, "update_memory"):
            agent.update_memory(state, step)

    def test_hls_prompt_has_rag_context(self):
        agent = load_agent(self.substrate_name, self.agent_type, self.llm_type, agent_id="player_0")
        state = make_minimal_state("player_0")
        step = 10

        self._prime_memory(agent, state, step)

        prompt = agent.generate_hls_user_message(state, step=step)

        # 1) まず「RAG markerが入っているか」（=RAG導入済みならここが通る）
        self.assertIn(
            self.marker,
            prompt,
            msg=(
                f"RAG marker '{self.marker}' not found in HLS prompt. "
                "=> RAGがpromptに組み込まれていない可能性が高いです。"
            ),
        )

        # 2) 生のdict dumpを貼り付けていないこと（RAG化できてるなら通常回避される）
        self.assertNotIn("{'yellow_box'", prompt, msg="Raw dict dump of memory_states is still in prompt")

    def test_feedback_prompt_has_rag_context(self):
        agent = load_agent(self.substrate_name, self.agent_type, self.llm_type, agent_id="player_0")
        state = make_minimal_state("player_0")
        step = 10

        self._prime_memory(agent, state, step)

        # ダミー引数（実装のシグネチャに合わせる）
        execution_outcomes = []                # 例: [{"subgoal":..., "ok":...}, ...]
        get_action_from_response_errors = []   # 例: ["json parse failed", ...]
        rewards = []                           # 例: [0.0] とかでもOK

        prompt = agent.generate_feedback_user_message(
            state,
            step,
            execution_outcomes,
            get_action_from_response_errors,
            rewards,
        )

        self.assertIn(
            self.marker,
            prompt,
            msg=(
                f"RAG marker '{self.marker}' not found in feedback prompt. "
                "=> RAGがpromptに組み込まれていない可能性が高いです。"
            ),
        )
        self.assertNotIn("{'yellow_box'", prompt, msg="Raw dict dump of memory_states is still in feedback prompt")


if __name__ == "__main__":
    unittest.main()
