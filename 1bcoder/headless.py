#!/usr/bin/env python3
"""1bcoder headless runner — executes a plan file without the Textual UI."""

import os
import re
import json
import subprocess
import concurrent.futures

import requests

# shared utilities and constants come from chat.py
from chat import (
    PLANS_DIR, BCODER_DIR,
    FIX_SYSTEM, PATCH_SYSTEM,
    read_file, edit_line, ai_fix,
    _parse_patch, _find_in_lines, _extract_code_block,
    _next_suffix_path, _load_plan, _save_plan,
    list_models,
)


class HeadlessRunner:
    def __init__(self, plan_path: str, base_url: str, model: str):
        self.plan_path = plan_path
        self.base_url  = base_url
        self.model     = model
        self.messages: list = []
        self.last_reply     = ""

    # ── public entry point ─────────────────────────────────────────────────────

    def run(self):
        lines   = _load_plan(self.plan_path)
        pending = [(i, l.rstrip("\n")) for i, l in enumerate(lines)
                   if not l.startswith("[v]")]
        if not pending:
            print("[headless] nothing to run — all steps already done")
            return
        name = os.path.basename(self.plan_path)
        print(f"[headless] {name} — {len(pending)} step(s)")
        for idx, cmd in pending:
            self._run_step(idx, cmd)
        print("\n[headless] done")

    # ── step execution ─────────────────────────────────────────────────────────

    def _run_step(self, idx: int, cmd: str):
        print(f"\n─── Step ────────────────────────────────")
        print(cmd)
        self._mark_done(idx)
        if   cmd.startswith("/read"):     self._do_read(cmd)
        elif cmd.startswith("/run"):      self._do_run(cmd)
        elif cmd.startswith("/host"):     self._do_host(cmd)
        elif cmd.startswith("/model"):    self._do_model(cmd)
        elif cmd.startswith("/fix"):      self._do_fix(cmd)
        elif cmd.startswith("/patch"):    self._do_patch(cmd)
        elif cmd.startswith("/save"):     self._do_save(cmd)
        elif cmd.startswith("/parallel"): self._do_parallel(cmd)
        elif cmd.startswith("/clear"):
            self.messages.clear()
            print("[context cleared]")
        elif not cmd.startswith("/"):
            self._do_chat(cmd)
        else:
            print(f"[headless] skipping unsupported command: {cmd}")

    def _mark_done(self, idx: int):
        lines = _load_plan(self.plan_path)
        lines[idx] = f"[v] {lines[idx]}"
        _save_plan(lines, self.plan_path)

    # ── command implementations ────────────────────────────────────────────────

    def _do_chat(self, text: str):
        self.messages.append({"role": "user", "content": text})
        print(f"[{self.model}] thinking...")
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json={"model": self.model, "messages": self.messages, "stream": False},
                timeout=300,
            )
            resp.raise_for_status()
            reply = resp.json().get("message", {}).get("content", "")
        except requests.exceptions.RequestException as e:
            print(f"error: {e}")
            self.messages.pop()
            return
        self.last_reply = reply
        self.messages.append({"role": "assistant", "content": reply})
        print(f"\n─── AI ──────────────────────────────────")
        print(reply)

    def _do_read(self, cmd: str):
        parts = cmd.split(None, 2)
        path  = parts[1] if len(parts) > 1 else ""
        start = end = None
        if len(parts) >= 3:
            try:
                s, e = parts[2].split("-")
                start, end = int(s), int(e)
            except ValueError:
                pass
        if not path:
            print("usage: /read <file> [start-end]")
            return
        try:
            content, total = read_file(path, start, end)
            label = path + (f" lines {start}-{end}" if start else f" ({total} lines)")
            self.messages.append({"role": "user",
                                   "content": f"[file: {label}]\n{content}"})
            print(f"[read {label}]")
        except (FileNotFoundError, OSError) as e:
            print(f"error: {e}")

    def _do_run(self, cmd: str):
        parts = cmd.split(None, 1)
        if len(parts) < 2:
            print("usage: /run <command>")
            return
        shell_cmd = parts[1]
        print(f"[run] {shell_cmd}")
        result = subprocess.run(shell_cmd, shell=True,
                                capture_output=True, text=True)
        output = (result.stdout + result.stderr).strip()
        if output:
            print(output)
        self.messages.append({"role": "user",
                               "content": f"[run: {shell_cmd}]\n{output}"})

    def _do_host(self, cmd: str):
        parts = cmd.split(None, 1)
        url = parts[1].strip().rstrip("/") if len(parts) > 1 else ""
        if not url:
            return
        if not url.startswith(("http://", "https://")):
            url = "http://" + url
        try:
            models = list_models(url)
            self.base_url = url
            self.model    = models[0]
            self.messages.clear()
            print(f"[host: {url}, model: {self.model}, context cleared]")
        except Exception as e:
            print(f"[host error: {e}]")

    def _do_model(self, cmd: str):
        parts = cmd.split(None, 1)
        if len(parts) == 2:
            self.model = parts[1].strip()
            print(f"[model: {self.model}]")

    def _do_fix(self, cmd: str):
        parts = cmd[4:].strip().split(None, 2)
        path  = parts[0] if parts else ""
        start = end = None
        hint  = ""
        if not path:
            print("usage: /fix <file> [start-end] [hint]")
            return
        if len(parts) >= 2:
            if re.match(r'^\d+-\d+$', parts[1]):
                s, e   = parts[1].split("-")
                start, end = int(s), int(e)
                hint   = parts[2] if len(parts) >= 3 else ""
            else:
                hint = " ".join(parts[1:])
        try:
            content, total = read_file(path, start, end)
            label = path + (f" lines {start}-{end}" if start else f" ({total} lines)")
        except (FileNotFoundError, OSError) as e:
            print(f"error: {e}")
            return
        lineno, new_content = ai_fix(self.base_url, self.model, content, label, hint)
        if lineno is None:
            print("[fix] could not parse LINE N: response")
            return
        try:
            edit_line(path, lineno, new_content)
            print(f"[fix] {path} line {lineno}: {new_content}")
        except (ValueError, OSError) as e:
            print(f"error: {e}")

    def _do_patch(self, cmd: str):
        parts = cmd[6:].strip().split(None, 2)
        path  = parts[0] if parts else ""
        start = end = None
        hint  = ""
        if not path:
            print("usage: /patch <file> [start-end] [hint]")
            return
        if len(parts) >= 2:
            if re.match(r'^\d+-\d+$', parts[1]):
                s, e   = parts[1].split("-")
                start, end = int(s), int(e)
                hint   = parts[2] if len(parts) >= 3 else ""
            else:
                hint = " ".join(parts[1:])
        try:
            content, total = read_file(path, start, end)
            label = path + (f" lines {start}-{end}" if start else f" ({total} lines)")
        except (FileNotFoundError, OSError) as e:
            print(f"error: {e}")
            return
        user_msg = f"Fix the code in this file ({label}):\n```\n{content}```"
        if hint:
            user_msg = f"{hint}\n\n{user_msg}"
        msgs = [
            {"role": "system", "content": PATCH_SYSTEM},
            {"role": "user",   "content": user_msg},
        ]
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json={"model": self.model, "messages": msgs, "stream": False},
                timeout=120,
            )
            resp.raise_for_status()
            raw = resp.json().get("message", {}).get("content", "")
        except requests.exceptions.RequestException as e:
            print(f"error: {e}")
            return
        search_text, replace_text = _parse_patch(raw)
        if search_text is None:
            print("[patch] could not parse SEARCH/REPLACE block")
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except (FileNotFoundError, OSError) as e:
            print(f"error: {e}")
            return
        si, ei = _find_in_lines(lines, search_text)
        if si is None:
            print("[patch] SEARCH text not found in file")
            return
        replace_lines = replace_text.splitlines(keepends=True)
        if replace_lines and not replace_lines[-1].endswith("\n"):
            replace_lines[-1] += "\n"
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines[:si] + replace_lines + lines[ei:])
        print(f"[patch] {path}: lines {si+1}–{ei} replaced")

    def _do_save(self, cmd: str):
        parts = cmd.split(None, 2)
        path  = parts[1] if len(parts) > 1 else ""
        mode  = parts[2].strip() if len(parts) > 2 else "overwrite"
        if not path:
            print("usage: /save <file> [mode]")
            return
        if not self.last_reply:
            print("[save] no AI reply to save yet")
            return
        content = _extract_code_block(self.last_reply) if mode == "code" else self.last_reply
        try:
            if mode == "append_below":
                with open(path, "a", encoding="utf-8") as f:
                    f.write("\n" + content)
            elif mode == "append_above":
                existing = open(path, encoding="utf-8").read() if os.path.exists(path) else ""
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content + "\n" + existing)
            elif mode == "add_suffix":
                path = _next_suffix_path(path)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
            else:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
            print(f"[saved: {path}]")
        except OSError as e:
            print(f"error: {e}")

    def _do_parallel(self, cmd: str):
        tokens = cmd.split()[1:]
        workers, prompt_parts = [], []
        for token in tokens:
            if "|" in token:
                parts = token.split("|", 2)
                if len(parts) == 3:
                    workers.append(tuple(parts))
                else:
                    print(f"[parallel] bad spec (need host|model|file): {token}")
                    return
            else:
                prompt_parts.append(token)
        if not workers:
            print("usage: /parallel <prompt> host:port|model|file ...")
            return
        prompt   = " ".join(prompt_parts)
        messages = list(self.messages)
        if prompt:
            messages = messages + [{"role": "user", "content": prompt}]
        if not messages:
            print("[parallel] no prompt and no context")
            return
        print(f"[parallel] sending to {len(workers)} model(s)...")

        def call_one(host, model, filename):
            url = host if host.startswith("http") else f"http://{host}"
            try:
                resp = requests.post(
                    f"{url}/api/chat",
                    json={"model": model, "messages": messages, "stream": False},
                    timeout=300,
                )
                resp.raise_for_status()
                reply = resp.json().get("message", {}).get("content", "")
            except Exception as e:
                return host, model, filename, None, str(e)
            dirpart = os.path.dirname(filename)
            if dirpart:
                os.makedirs(dirpart, exist_ok=True)
            with open(filename, "w", encoding="utf-8") as f:
                f.write(reply)
            return host, model, filename, reply, None

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(workers)) as pool:
            futs = {pool.submit(call_one, h, m, f): (h, m, f) for h, m, f in workers}
            for fut in concurrent.futures.as_completed(futs):
                host, model, filename, reply, err = fut.result()
                if err:
                    print(f"[parallel] {model}@{host} — error: {err}")
                else:
                    print(f"[parallel] {model}@{host} → {filename} ({len(reply)} chars)")
        print("[parallel] done")
