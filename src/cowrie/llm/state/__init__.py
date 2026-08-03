# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: Transactional world state. Attacker input is parsed into
# ABOUTME: intents (cmd_parser), validated against the world here, and
# ABOUTME: committed only if the command is allowed to succeed. Exists
# ABOUTME: because mutations used to be applied before anyone knew whether
# ABOUTME: the command worked, so a refused write still changed the world.

from cowrie.llm.state.planner import Plan, plan_mutations

__all__ = ["Plan", "plan_mutations"]
