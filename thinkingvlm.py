import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import AutoProcessor, Glm4vForConditionalGeneration

from goal_lunarlander_env import make_goal_lunarlander_env, sanitize_action


@dataclass
class VlmDecision:
    action: np.ndarray
    raw_text: str
    parsed_ok: bool


def _now_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


class _StopOnGeneratedJsonEnd(StoppingCriteria):
    """
    Stop generation as soon as the *generated* portion contains a '}' after a '{' appeared.

    Important: we MUST ignore braces in the prompt (the prompt itself contains {"action":[...]}),
    otherwise generation will stop immediately after the first token (often "<think>").
    """

    def __init__(self, tokenizer, start_length: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.start_length = int(start_length)
        self._started = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Only look at newly generated tokens (ignore the prompt).
        gen = input_ids[0, self.start_length :]
        if gen.numel() == 0:
            return False

        # Decode only the last ~128 generated tokens to keep it cheap.
        tail = gen[-128:].tolist()
        txt = self.tokenizer.decode(tail, skip_special_tokens=True)

        if "{" in txt:
            self._started = True
        if self._started and "}" in txt:
            return True
        return False


def _build_prompt(
    goal_x: float,
    step_idx: int,
    episode_idx: int,
    last_action: np.ndarray | None,
    last_reward: float | None,
    last_info: dict[str, Any] | None,
) -> str:
    """
    Prompt策略（高约束、可解析）：
    - 强制输出 JSON，字段固定，动作范围明确
    - 不要求“解释”，避免模型输出太长/跑题（但保留一个短 reason 方便 debug）
    """
    goal_side = "LEFT" if goal_x < 0 else "RIGHT"
    return f"""You control LunarLanderContinuous (continuous).

TASK: land on the pad near goal_x.
LEFT/RIGHT (do NOT swap):
- Right side of image = +x, left side of image = -x.
- goal_x < 0 => go LEFT.  goal_x > 0 => go RIGHT.

episode={episode_idx} step={step_idx} goal_x={goal_x:.3f} goal_side={goal_side}

Action a=[main,lateral], each in [-1,1]:
- main < 0: main engine OFF; 0..1: main thrust (50%..100%).
- |lateral| < 0.5: no lateral; lateral <= -0.5: LEFT booster; lateral >= 0.5: RIGHT booster (50%..100%).
IMPORTANT movement mapping:
- To move RIGHT (+x): use lateral <= -0.5 (LEFT booster).
- To move LEFT  (-x): use lateral >= 0.5 (RIGHT booster).

OUTPUT RULES (strict):
- Do NOT output any thinking, analysis, or <think> tags.
- Output ONE single-line JSON only.
- Start immediately with '{{' (no leading text).

{{"action":[main,lateral]}}
"""


def _extract_action_from_text(text: str) -> tuple[np.ndarray | None, bool]:
    """
    Robust parsing:
    - Prefer JSON with key "action"
    - Fallback: first bracketed pair like [a, b]
    """
    # Drop <think> blocks (both closed and unclosed) to improve parse robustness
    if "<think>" in text:
        # Remove closed blocks
        text = re.sub(r"<think>[\s\S]*?</think>", "", text)
        # Remove leading unclosed think
        if "<think>" in text and "{" in text:
            text = text[text.find("{") :]

    # 1) JSON parse attempt (find first {...})
    try:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "action" in obj:
                act = obj["action"]
                if (
                    isinstance(act, (list, tuple))
                    and len(act) == 2
                    and all(isinstance(x, (int, float)) for x in act)
                ):
                    return np.asarray([float(act[0]), float(act[1])], dtype=np.float32), True
    except Exception:
        pass

    # 2) Action field regex (works even if braces/quotes are messed up)
    try:
        m1 = re.search(
            r'"action"\s*:\s*\[\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\]',
            text,
        )
        if m1:
            a0 = float(m1.group(1))
            a1 = float(m1.group(2))
            return np.asarray([a0, a1], dtype=np.float32), True
    except Exception:
        pass

    # 3) Fallback: first bracketed pair like [a, b]
    try:
        m2 = re.search(
            r"\[\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\]",
            text,
        )
        if m2:
            a0 = float(m2.group(1))
            a1 = float(m2.group(2))
            return np.asarray([a0, a1], dtype=np.float32), True
    except Exception:
        pass

    return None, False


@torch.inference_mode()
def vlm_choose_action(
    model: Glm4vForConditionalGeneration,
    processor: AutoProcessor,
    image: Image.Image,
    prompt_text: str,
    *,
    max_new_tokens: int,
    temperature: float,
) -> VlmDecision:
    """
    使用 chat template，把“图 + 文本”喂给模型，强制它输出 JSON 动作。
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    # 常见多模态 processor 用法：先把 messages -> text（包含 <image> 占位），再和 images 一起编码
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

    start_len = int(inputs["input_ids"].shape[1])
    stopping = StoppingCriteriaList([_StopOnGeneratedJsonEnd(processor.tokenizer, start_len)])
    generated = model.generate(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        do_sample=float(temperature) > 0.0,
        temperature=float(max(1e-6, temperature)),
        stopping_criteria=stopping,
    )
    # 只解码新生成部分
    gen_ids = generated[0][inputs["input_ids"].shape[1] :]
    out_text = processor.decode(gen_ids, skip_special_tokens=True)

    action, ok = _extract_action_from_text(out_text)
    if action is None:
        action = np.zeros(2, dtype=np.float32)
    action = np.clip(action.astype(np.float32), -1.0, 1.0)
    return VlmDecision(action=action, raw_text=out_text, parsed_ok=ok)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="LunarLanderContinuous-v3")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--manual_seed", type=int, default=1)

    # goal env config
    parser.add_argument("--goal_left", type=float, default=-0.2)
    parser.add_argument("--goal_right", type=float, default=0.2)
    parser.add_argument("--goal_reward_weight", type=float, default=1.0)
    parser.add_argument("--goal_tolerance", type=float, default=0.10)
    parser.add_argument("--goal_success_reward", type=float, default=100.0)
    parser.add_argument("--goal_fail_landed_reward", type=float, default=-100.0)

    # VLM config
    parser.add_argument("--model_path", type=str, default="zai-org/GLM-4.1V-9B-Thinking")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument(
        "--vlm_retries",
        type=int,
        default=2,
        help="If output is not parseable JSON, retry this many times with stricter instruction.",
    )
    parser.add_argument(
        "--vlm_every",
        type=int,
        default=5,
        help="Call VLM every N env steps (reduce cost/latency).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional sleep seconds per step for visualization pacing.",
    )
    parser.add_argument(
        "--debug_print",
        action="store_true",
        help="Print VLM raw output and parse status occasionally.",
    )
    # Trajectory logging
    parser.add_argument(
        "--save_log",
        action="store_true",
        help="Save per-episode JSONL logs (goal_x, actions, rewards, parse status, raw model output).",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="vlm_logs",
        help="Directory to store trajectory logs.",
    )
    # GIF recording
    parser.add_argument(
        "--save_gif",
        action="store_true",
        help="Save a GIF for each episode.",
    )
    parser.add_argument(
        "--gif_dir",
        type=str,
        default="gifs",
        help="Directory to store episode GIFs.",
    )
    parser.add_argument(
        "--gif_fps",
        type=float,
        default=20.0,
        help="Playback FPS for saved GIFs.",
    )
    parser.add_argument(
        "--gif_every",
        type=int,
        default=2,
        help="Save one frame every N env steps (reduce GIF size).",
    )
    args = parser.parse_args()

    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)

    run_id = _now_run_id()
    run_dir = os.path.join(args.log_dir, run_id) if args.save_log else None
    if args.save_log and run_dir is not None:
        os.makedirs(run_dir, exist_ok=True)

    env = make_goal_lunarlander_env(
        env_name=args.env_name,
        goals=(args.goal_left, args.goal_right),
        goal_reward_weight=args.goal_reward_weight,
        goal_tolerance=args.goal_tolerance,
        goal_success_reward=args.goal_success_reward,
        goal_fail_landed_reward=args.goal_fail_landed_reward,
        render_mode="rgb_array",
    )

    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=True)
    model = Glm4vForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    action_dim = int(env.action_space.shape[0])
    max_action = float(env.action_space.high[0])

    all_episode_returns: list[float] = []

    for ep in range(int(args.episodes)):
        obs, info = env.reset(seed=args.manual_seed + ep)
        goal_x = float(info.get("goal_x", 0.0))

        ep_return = 0.0
        last_action = None
        last_reward = None
        last_info = None

        frames: list[Image.Image] = []
        if args.save_gif:
            os.makedirs(args.gif_dir, exist_ok=True)

        log_f = None
        if args.save_log and run_dir is not None:
            # One jsonl per episode for easy streaming/replay
            log_path = os.path.join(run_dir, f"episode_{ep}_goal{goal_x:+.3f}.jsonl")
            log_f = open(log_path, "w", encoding="utf-8")
            meta = {
                "type": "episode_start",
                "episode": ep,
                "goal_x": goal_x,
                "env_name": args.env_name,
                "seed": args.manual_seed + ep,
                "vlm_every": int(max(1, args.vlm_every)),
            }
            log_f.write(json.dumps(meta, ensure_ascii=False) + "\n")

        # 初始化图像（有些 env reset 后要先 render 一次才有画面）
        frame = env.render()
        if frame is None:
            # 保险：step 之前可能拿不到图
            frame = np.zeros((400, 600, 3), dtype=np.uint8)
        image = Image.fromarray(frame)
        if args.save_gif:
            frames.append(image.copy())

        # 当前保持的动作（当 vlm_every>1 时复用）
        held_action = np.zeros(action_dim, dtype=np.float32)
        last_decision: VlmDecision | None = None
        last_valid_action = np.zeros(action_dim, dtype=np.float32)

        for t in range(int(args.max_steps)):
            if t % int(max(1, args.vlm_every)) == 0:
                # Retry loop: if model rambles (<think>/no JSON), treat as invalid and reprompt stricter.
                decision = None
                for attempt in range(int(max(1, args.vlm_retries)) + 1):
                    prompt_text = _build_prompt(
                        goal_x=goal_x,
                        step_idx=t,
                        episode_idx=ep,
                        last_action=last_action,
                        last_reward=last_reward,
                        last_info=last_info,
                    )
                    if attempt > 0:
                        prompt_text += "\nREMINDER: Output ONLY JSON. No extra words. Start with '{'.\n"
                    decision = vlm_choose_action(
                        model,
                        processor,
                        image,
                        prompt_text,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                    )
                    if decision.parsed_ok:
                        break

                assert decision is not None
                last_decision = decision
                if decision.parsed_ok:
                    held_action = decision.action
                    last_valid_action = decision.action
                else:
                    # Strict policy: invalid output => reuse last valid action (or [0,0] at start).
                    held_action = last_valid_action.copy()
                if args.debug_print and (t % 20 == 0):
                    print(
                        f"[ep={ep} t={t}] parsed_ok={decision.parsed_ok} action={held_action.tolist()}"
                    )
                    print(f"raw:\n{decision.raw_text}\n")

            # Gym expects action shape/dtype; also clip by env max_action
            action = sanitize_action(held_action, action_dim=action_dim, max_action=max_action)

            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)

            last_action = action
            last_reward = float(reward)
            last_info = dict(info)

            if log_f is not None:
                rec = {
                    "type": "step",
                    "episode": ep,
                    "t": t,
                    "goal_x": goal_x,
                    "goal_side": "LEFT" if goal_x < 0 else "RIGHT",
                    "action": [float(action[0]), float(action[1])],
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "goal_dx": float(last_info.get("goal_dx", 0.0)),
                    "goal_success": bool(last_info.get("goal_success", False)),
                    "vlm_called": bool(t % int(max(1, args.vlm_every)) == 0),
                    "parsed_ok": (bool(last_decision.parsed_ok) if last_decision else None),
                    "raw_vlm": (last_decision.raw_text if last_decision else None),
                }
                log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            frame = env.render()
            if frame is not None:
                image = Image.fromarray(frame)
                if args.save_gif and (t % int(max(1, args.gif_every)) == 0):
                    frames.append(image.copy())

            if args.sleep > 0:
                time.sleep(float(args.sleep))

            if terminated or truncated:
                break

        all_episode_returns.append(ep_return)
        goal_success = bool(last_info.get("goal_success", False)) if last_info else False
        print(
            f"Episode {ep}: return={ep_return:.2f}, goal_x={goal_x:.3f}, goal_success={goal_success}"
        )
        if log_f is not None:
            end = {
                "type": "episode_end",
                "episode": ep,
                "goal_x": goal_x,
                "return": float(ep_return),
                "goal_success": bool(goal_success),
            }
            log_f.write(json.dumps(end, ensure_ascii=False) + "\n")
            log_f.close()
        if args.save_gif and frames:
            duration_ms = int(1000.0 / float(max(1e-6, args.gif_fps)))
            out_path = os.path.join(args.gif_dir, f"episode_{ep}_goal{goal_x:+.3f}.gif")
            frames[0].save(
                out_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration_ms,
                loop=0,
                optimize=False,
            )

    env.close()
    mean_return = float(np.mean(all_episode_returns)) if all_episode_returns else 0.0
    print(f"Done. Episodes={len(all_episode_returns)}, mean_return={mean_return:.2f}")


if __name__ == "__main__":
    main()

