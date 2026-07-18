"""
app.py
------
Campus Helpdesk Chatbot — a rule-based AI chatbot with an ANIMATED Tkinter GUI.

Animations (built with Tkinter's built-in `after()` loop — no extra
libraries needed):
    1. A bouncing 3-dot "Bot is typing..." indicator while the bot "thinks"
    2. A fade-in effect on every new bot message (text color animates in)
    3. A quick color-flash on the Send button when clicked

Ties together:
    core/chatbot.py   -> problem setup + core AI logic (rule matching)
    core/explain.py   -> explainability module
    core/evaluate.py  -> evaluation module (accuracy on a labeled test set)

Run with:  python app.py
"""

import os
import math
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from core.chatbot import load_data, run_model_or_algorithm
from core.explain import generate_explanation
from core.evaluate import load_test_set, evaluate_chatbot

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTENTS_PATH = os.path.join(BASE_DIR, "data", "intents.json")
TEST_SET_PATH = os.path.join(BASE_DIR, "data", "test_questions.json")

BG = "#f2f4f7"
CHAT_BG = "#ffffff"
USER_BUBBLE = "#1976d2"
BOT_TEXT_FINAL = "#1a1a1a"
SEND_BLUE = "#1976d2"
SEND_FLASH = "#0d47a1"

THINKING_DELAY_MS = 650   # how long the typing animation plays before the reply appears
FADE_STEPS = 10
FADE_INTERVAL_MS = 35


def _interpolate_color(c1, c2, ratio):
    """Blend two '#rrggbb' colors; ratio 0 -> c1, ratio 1 -> c2."""
    c1 = c1.lstrip("#")
    c2 = c2.lstrip("#")
    r1, g1, b1 = int(c1[0:2], 16), int(c1[2:4], 16), int(c1[4:6], 16)
    r2, g2, b2 = int(c2[0:2], 16), int(c2[2:4], 16), int(c2[4:6], 16)
    r = int(r1 + (r2 - r1) * ratio)
    g = int(g1 + (g2 - g1) * ratio)
    b = int(b1 + (b2 - b1) * ratio)
    return f"#{r:02x}{g:02x}{b:02x}"


class TypingIndicator:
    """A small canvas with 3 dots that bounce in a sine-wave pattern,
    shown while the bot is 'thinking' and hidden once the reply is ready."""

    def __init__(self, parent, bg=CHAT_BG):
        self.canvas = tk.Canvas(parent, width=70, height=28, bg=bg, highlightthickness=0)
        self.dots = [self.canvas.create_oval(0, 0, 0, 0, fill="#9aa0a6", outline="") for _ in range(3)]
        self._base_y = 14
        self._running = False
        self._frame = 0
        self._job = None

    def start(self):
        self.canvas.pack(anchor="w", padx=10, pady=(4, 4))
        self._running = True
        self._frame = 0
        self._animate()

    def stop(self):
        self._running = False
        if self._job is not None:
            self.canvas.after_cancel(self._job)
            self._job = None
        self.canvas.pack_forget()

    def _animate(self):
        if not self._running:
            return
        for i, dot in enumerate(self.dots):
            phase = self._frame * 0.35 + i * 1.3
            offset = math.sin(phase) * 5
            cx = 12 + i * 20
            cy = self._base_y + offset
            r = 4
            self.canvas.coords(dot, cx - r, cy - r, cx + r, cy + r)
        self._frame += 1
        self._job = self.canvas.after(55, self._animate)


class ChatbotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Campus Helpdesk Chatbot — Rule-Based AI (Animated GUI)")
        self.root.geometry("900x660")
        self.root.configure(bg=BG)

        self._msg_counter = 0

        # --------------------------------------------------------------
        # A) PROBLEM SETUP MODULE — load the rules dataset once at start
        # --------------------------------------------------------------
        try:
            self.intents_data = load_data(INTENTS_PATH)
            self.data = {"intents_data": self.intents_data}
            self.load_error = None
        except Exception as e:
            self.intents_data = None
            self.data = None
            self.load_error = str(e)

        self.last_result = None
        self.last_explanation = None
        self.chart_canvas_widget = None
        self._chart_placeholder = None

        self._build_layout()

        if self.load_error:
            self._set_status(f"⚠️ Failed to load intents data: {self.load_error}", error=True)
        else:
            self._append_bot_message(
                "Hello! I'm the Campus Helpdesk bot. Ask me about admissions, "
                "courses, fees, timings, or the library."
            )

    # ------------------------------------------------------------------
    # UI construction (render_ui)
    # ------------------------------------------------------------------
    def _build_layout(self):
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=12, pady=(10, 0))
        tk.Label(header, text="🎓 Campus Helpdesk Chatbot", bg=BG,
                 font=("Segoe UI", 16, "bold")).pack(side="left")

        self.status_var = tk.StringVar(value="Ready")
        self.status_label = tk.Label(header, textvariable=self.status_var, bg=BG,
                                      fg="#2e7d32", font=("Segoe UI", 10))
        self.status_label.pack(side="right")

        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True, padx=12, pady=10)

        left = tk.Frame(main, bg=BG)
        left.pack(side="left", fill="both", expand=True)

        right = tk.Frame(main, bg=BG, width=300)
        right.pack(side="right", fill="y", padx=(10, 0))
        right.pack_propagate(False)

        # ---- C) VISUAL UI MODULE: chat window ----
        chat_frame = tk.Frame(left, bg=CHAT_BG, bd=1, relief="solid")
        chat_frame.pack(fill="both", expand=True)

        self.chat_canvas = tk.Text(chat_frame, wrap="word", bg=CHAT_BG,
                                    font=("Segoe UI", 10), state="disabled",
                                    padx=10, pady=10, bd=0)
        self.chat_canvas.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(chat_frame, command=self.chat_canvas.yview)
        scrollbar.pack(side="right", fill="y")
        self.chat_canvas.configure(yscrollcommand=scrollbar.set)

        self.chat_canvas.tag_configure("user", justify="right", foreground=USER_BUBBLE,
                                        font=("Segoe UI", 10, "bold"), spacing1=6, spacing3=2)
        self.chat_canvas.tag_configure("meta", justify="left", foreground="#8a8f98",
                                        font=("Segoe UI", 8, "italic"), spacing3=4)

        # Typing indicator lives just below the chat log, inside `left`
        self.typing_indicator = TypingIndicator(left, bg=BG)

        # ---- Controls: entry + buttons ----
        controls = tk.Frame(left, bg=BG)
        controls.pack(fill="x", pady=(8, 0))

        self.entry_var = tk.StringVar()
        self.entry = tk.Entry(controls, textvariable=self.entry_var, font=("Segoe UI", 11))
        self.entry.pack(side="left", fill="x", expand=True, ipady=6, padx=(0, 8))
        self.entry.bind("<Return>", lambda e: self._on_send())
        self.entry.focus_set()

        self.send_btn = tk.Button(controls, text="Send ➤", command=self._on_send, bg=SEND_BLUE,
                                   fg="white", font=("Segoe UI", 10, "bold"), relief="flat",
                                   padx=14, pady=6, activebackground=SEND_FLASH, cursor="hand2")
        self.send_btn.pack(side="left")

        button_row = tk.Frame(left, bg=BG)
        button_row.pack(fill="x", pady=(8, 0))

        tk.Button(button_row, text="🧹 Clear Chat", command=self._on_clear,
                  relief="flat", bg="#eeeeee", padx=10, pady=4, cursor="hand2").pack(side="left")
        tk.Button(button_row, text="📊 Run Evaluation", command=self._on_evaluate,
                  relief="flat", bg="#eeeeee", padx=10, pady=4, cursor="hand2").pack(side="left", padx=6)

        # ---- Right panel: Result + Explainability (D & E modules) ----
        tk.Label(right, text="📋 Result Panel", bg=BG, font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.result_box = tk.Text(right, height=6, wrap="word", bg="white", bd=1,
                                   relief="solid", font=("Segoe UI", 9), state="disabled")
        self.result_box.pack(fill="x", pady=(4, 12))

        tk.Label(right, text="💡 Why this answer?", bg=BG, font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.explain_box = tk.Text(right, height=8, wrap="word", bg="white", bd=1,
                                    relief="solid", font=("Segoe UI", 9), state="disabled")
        self.explain_box.pack(fill="x", pady=(4, 12))

        tk.Label(right, text="📈 Evaluation Chart", bg=BG, font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.chart_frame = tk.Frame(right, bg="white", bd=1, relief="solid", height=220)
        self.chart_frame.pack(fill="both", expand=True, pady=(4, 0))
        self._show_placeholder_chart_message()

    # ------------------------------------------------------------------
    # Chat helpers
    # ------------------------------------------------------------------
    def _append_user_message(self, text):
        self.chat_canvas.configure(state="normal")
        self.chat_canvas.insert("end", f"You  {datetime.now().strftime('%H:%M')}\n", "meta")
        self.chat_canvas.insert("end", f"{text}\n", "user")
        self.chat_canvas.configure(state="disabled")
        self.chat_canvas.see("end")

    def _append_bot_message(self, text):
        """Insert a bot message and animate it fading in from light to dark text."""
        tag_name = f"botmsg_{self._msg_counter}"
        self._msg_counter += 1

        self.chat_canvas.configure(state="normal")
        self.chat_canvas.insert("end", f"Bot  {datetime.now().strftime('%H:%M')}\n", "meta")
        self.chat_canvas.insert("end", f"{text}\n", tag_name)
        self.chat_canvas.tag_configure(tag_name, justify="left", foreground=CHAT_BG,
                                        font=("Segoe UI", 10), spacing3=10)
        self.chat_canvas.configure(state="disabled")
        self.chat_canvas.see("end")

        self._fade_in_tag(tag_name, step=0)

    def _fade_in_tag(self, tag_name, step):
        """Animate a text tag's foreground color from the background color to
        the final text color over FADE_STEPS frames — a simple fade-in effect."""
        if step > FADE_STEPS:
            self.chat_canvas.tag_configure(tag_name, foreground=BOT_TEXT_FINAL)
            return
        ratio = step / FADE_STEPS
        color = _interpolate_color(CHAT_BG, BOT_TEXT_FINAL, ratio)
        self.chat_canvas.tag_configure(tag_name, foreground=color)
        self.root.after(FADE_INTERVAL_MS, lambda: self._fade_in_tag(tag_name, step + 1))

    def _set_status(self, text, error=False):
        self.status_var.set(text)
        self.status_label.configure(fg="#d32f2f" if error else "#2e7d32")

    def _flash_send_button(self):
        """Quick color flash animation on the Send button for tactile feedback."""
        self.send_btn.configure(bg=SEND_FLASH)
        self.root.after(150, lambda: self.send_btn.configure(bg=SEND_BLUE))

    # ------------------------------------------------------------------
    # B) CORE LOGIC — send button handler
    # ------------------------------------------------------------------
    def _on_send(self):
        if self.data is None:
            messagebox.showerror("Data error", f"Chatbot data failed to load:\n{self.load_error}")
            return

        message = self.entry_var.get()
        self._flash_send_button()

        if not message.strip():
            self._set_status("⚠️ Please type a message before sending.", error=True)
            return

        self._append_user_message(message)
        self.entry_var.set("")
        self._set_status("🤔 Bot is typing...")
        self.entry.configure(state="disabled")
        self.send_btn.configure(state="disabled")

        # Start the bouncing-dots animation, then reveal the reply after a
        # short delay so the animation is actually visible (classic chat UX).
        self.typing_indicator.start()
        self.root.after(THINKING_DELAY_MS, lambda: self._process_and_respond(message))

    def _process_and_respond(self, message):
        self.typing_indicator.stop()
        self.entry.configure(state="normal")
        self.send_btn.configure(state="normal")
        self.entry.focus_set()

        try:
            result = run_model_or_algorithm(self.data, {"message": message})
        except ValueError as e:
            self._set_status(f"⚠️ {e}", error=True)
            return

        self.last_result = result
        self._append_bot_message(result["response"])
        self._update_result_panel(result)

        explanation = generate_explanation(result)
        self.last_explanation = explanation
        self._update_explain_panel(explanation)

        self._set_status(f"✅ Ready · last response in {result['runtime_ms']} ms")

    def _on_clear(self):
        self.chat_canvas.configure(state="normal")
        self.chat_canvas.delete("1.0", "end")
        self.chat_canvas.configure(state="disabled")
        self._set_status("Chat cleared.")

    # ------------------------------------------------------------------
    # D) EXPLAINABILITY + result panel rendering
    # ------------------------------------------------------------------
    def _update_result_panel(self, result):
        self.result_box.configure(state="normal")
        self.result_box.delete("1.0", "end")
        lines = [
            f"Intent: {result['tag']}",
            f"Confidence: {result['confidence']:.0%}",
            f"Fallback used: {'Yes' if result['is_fallback'] else 'No'}",
            f"Response time: {result['runtime_ms']} ms",
        ]
        self.result_box.insert("end", "\n".join(lines))
        self.result_box.configure(state="disabled")

    def _update_explain_panel(self, explanation):
        self.explain_box.configure(state="normal")
        self.explain_box.delete("1.0", "end")
        text = explanation["summary"] + "\n\n" + "\n".join(f"• {f}" for f in explanation["key_factors"])
        self.explain_box.insert("end", text)
        self.explain_box.configure(state="disabled")

    # ------------------------------------------------------------------
    # E) EVALUATION MODULE
    # ------------------------------------------------------------------
    def _show_placeholder_chart_message(self):
        label = tk.Label(self.chart_frame, text="Click 'Run Evaluation' to test\naccuracy on sample questions.",
                          bg="white", fg="#888888", font=("Segoe UI", 9), justify="center")
        label.pack(expand=True)
        self._chart_placeholder = label

    def _on_evaluate(self):
        if self.data is None:
            messagebox.showerror("Data error", f"Chatbot data failed to load:\n{self.load_error}")
            return

        self._set_status("📊 Running evaluation on test set...")
        self.root.update_idletasks()

        test_set = load_test_set(TEST_SET_PATH)
        eval_result = evaluate_chatbot(self.intents_data, test_set)

        self._render_evaluation_chart(eval_result)

        wrong = [d for d in eval_result["details"] if not d["correct"]]
        wrong_str = "\n".join(
            f"  - '{d['question']}' expected={d['expected']} got={d['predicted']}" for d in wrong
        ) or "  (none)"

        self._set_status(
            f"✅ Evaluation done: {eval_result['accuracy']:.0%} accuracy "
            f"({eval_result['correct']}/{eval_result['total']})"
        )

        messagebox.showinfo(
            "Evaluation Results",
            f"Accuracy: {eval_result['accuracy']:.1%} "
            f"({eval_result['correct']}/{eval_result['total']} correct)\n"
            f"Average response time: {eval_result['avg_runtime_ms']} ms\n\n"
            f"Misclassified questions:\n{wrong_str}"
        )

    def _render_evaluation_chart(self, eval_result):
        if self._chart_placeholder is not None:
            self._chart_placeholder.destroy()
            self._chart_placeholder = None
        if self.chart_canvas_widget is not None:
            self.chart_canvas_widget.get_tk_widget().destroy()

        tags = sorted(set(d["expected"] for d in eval_result["details"]))
        acc_per_tag = {}
        for tag in tags:
            tag_items = [d for d in eval_result["details"] if d["expected"] == tag]
            correct = sum(1 for d in tag_items if d["correct"])
            acc_per_tag[tag] = correct / len(tag_items) if tag_items else 0

        fig = Figure(figsize=(3.6, 3.2), dpi=90)
        ax = fig.add_subplot(111)
        ax.bar(list(acc_per_tag.keys()), [v * 100 for v in acc_per_tag.values()], color="#1976d2")
        ax.set_ylabel("Accuracy (%)", fontsize=8)
        ax.set_ylim(0, 110)
        ax.tick_params(axis="x", labelrotation=60, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_title(f"Overall: {eval_result['accuracy']:.0%}", fontsize=9, fontweight="bold")
        fig.tight_layout()

        self.chart_canvas_widget = FigureCanvasTkAgg(fig, master=self.chart_frame)
        self.chart_canvas_widget.draw()
        self.chart_canvas_widget.get_tk_widget().pack(fill="both", expand=True)


def render_ui():
    root = tk.Tk()
    ChatbotApp(root)
    root.mainloop()


if __name__ == "__main__":
    render_ui()
