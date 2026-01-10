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
        self.step = 10

        # RAG導入後にpromptへ必ず入れるマーカー（あなたが決める）
        self.marker = os.getenv("RAG_MARKER", "RAG_CONTEXT_BEGIN")
        self.require_rag = os.getenv("REQUIRE_RAG", "0") == "1"

    def _prime_memory(self, agent, state, step):
        if hasattr(agent, "update_memory"):
            agent.update_memory(state, step)

    def test_hls_prompt_contains_memory(self):
        """現状でも必ず通るべき：update_memory後に、memory内容がpromptに反映されること"""
        agent = load_agent(self.substrate_name, self.agent_type, self.llm_type, agent_id="player_0")
        state = make_minimal_state("player_0")
        self._prime_memory(agent, state, self.step)

        prompt = agent.generate_hls_user_message(state, step=self.step)

        self.assertIn("Retrieved memory context (RAG)", prompt)
        self.assertIn("RAG_CONTEXT_BEGIN", prompt)
        self.assertIn("RAG_CONTEXT_END", prompt)
        self.assertNotIn("Previously seen states from memory", prompt)
        self.assertNotIn("{'yellow_box'", prompt)


        # 観測した座標が入っている（memoryが空じゃないこと）
        self.assertIn("(5, 7)", prompt)
        self.assertIn("(9, 3)", prompt)

    def test_feedback_prompt_smoke(self):
        """落ちないことの確認（引数の型/順序のズレで落ちやすいので）"""
        agent = load_agent(self.substrate_name, self.agent_type, self.llm_type, agent_id="player_0")
        state = make_minimal_state("player_0")
        self._prime_memory(agent, state, self.step)

        execution_outcomes = []
        get_action_from_response_errors = []
        rewards = {"player_0": 0.0, "player_1": 0.0}

        prompt = agent.generate_feedback_user_message(
            state,
            execution_outcomes,
            get_action_from_response_errors,
            rewards,
            self.step,
        )
        self.assertIsInstance(prompt, str)
        self.assertIn("Strategy Request", prompt)

    def test_rag_marker_present_when_required(self):
        """
        REQUIRE_RAG=1 のときだけ厳密に判定：
        RAG導入済みなら marker がpromptに入るはず。
        """
        if not self.require_rag:
            self.skipTest("Set REQUIRE_RAG=1 to enforce RAG checks")

        agent = load_agent(self.substrate_name, self.agent_type, self.llm_type, agent_id="player_0")
        state = make_minimal_state("player_0")
        self._prime_memory(agent, state, self.step)

        prompt = agent.generate_hls_user_message(state, step=self.step)
        self.assertIn(self.marker, prompt, msg="RAG marker not found => RAGがpromptに組み込まれていません")

        # （任意）生のdict dumpを貼っていないこと
        self.assertNotIn("{'yellow_box'", prompt, msg="Raw dict dump is still present => retrievalに置換できていません")


if __name__ == "__main__":
    unittest.main()

