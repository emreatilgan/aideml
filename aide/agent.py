import shutil
import logging
import random
import time
from typing import Any, Callable, cast
from pathlib import Path
import re

import humanize
from .backend import FunctionSpec, compile_prompt_to_md, query
from .interpreter import ExecutionResult
from .journal import Journal, Node
from .utils import data_preview
from .utils.config import Config
from .utils.metric import MetricValue, WorstMetricValue
from .utils.response import extract_code, extract_text_up_to_code, wrap_code, trim_long_string

logger = logging.getLogger("aide")


def format_time(time_in_sec: int):
    return f"{time_in_sec // 3600}hrs {(time_in_sec % 3600) // 60}mins {time_in_sec % 60}secs"


ExecCallbackType = Callable[[str, bool], ExecutionResult]

review_func_spec = FunctionSpec(
    name="submit_review",
    json_schema={
        "type": "object",
        "properties": {
            "is_bug": {
                "type": "boolean",
                "description": "true if the output log shows that the execution failed or has some bug, otherwise false.",
            },
            "has_csv_submission": {
                "type": "boolean",
                "description": "true if the code saves the predictions on the test data"
                " in a `submission.csv` file in the `./submission/` directory, otherwise false."
                " Note that the file MUST be saved in the ./submission/ directory for this to be evaluated as true."
                " Otherwise, it should be evaluated as false."
                " You can assume the ./submission/ directory exists and is writable.",
            },
            "summary": {
                "type": "string",
                "description": "write a short summary (2-3 sentences) describing "
                " the empirical findings. Alternatively mention if there is a bug or"
                " the submission.csv was not properly produced."
                " DO NOT suggest fixes or improvements.",
            },
            "metric": {
                "type": "number",
                "description": "If the code ran successfully, report the value of the validation metric. For competitions evaluated by the mean of some metric (RMSE, RMSLE, Accuracy etc.) across multiple target columns, report the mean of some metric (RMSE, RMSLE, Accuracy etc.) averaged across all target columns (prefer a value clearly labeled 'mean_X' X being the column metric in the program output). Otherwise, leave it null.",
            },
            "lower_is_better": {
                "type": "boolean",
                "description": "true if the metric should be minimized (i.e. a lower metric value is better, such as with MSE), false if the metric should be maximized (i.e. a higher metric value is better, such as with accuracy).",
            },
        },
        "required": [
            "is_bug",
            "has_csv_submission",
            "summary",
            "metric",
            "lower_is_better",
        ],
    },
    description="Submit a review evaluating the output of the training script.",
)


class Agent:
    def __init__(
        self,
        task_desc: str,
        cfg: Config,
        journal: Journal,
    ):
        super().__init__()
        self.task_desc = task_desc
        self.cfg = cfg
        self.acfg = cfg.agent
        self.journal = journal
        self.data_preview: str | None = None
        self.start_time = time.time()
        self.current_step = 0
        # Stage 1 plan (cached between drafts to keep Stage 2 prompts compact)
        self.stage1_summary: str | None = None
        # Select exactly one draft among the first N to be guided by the knowledge base
        #self.guided_draft_index = (
        #    random.randint(0, self.acfg.search.num_drafts - 1)
        #    if self.acfg.search.num_drafts > 0
        #    else 0
        #)
        self.guided_draft_index = 0
        # Root path to the bundled knowledge base code examples
        self.kb_root = Path(__file__).parent / "knowledge_base" / "code_examples"

    def search_policy(self) -> Node | None:
        """Select a node to work on (or None to draft a new node)."""
        search_cfg = self.acfg.search

        # initial drafting
        if len(self.journal.draft_nodes) < search_cfg.num_drafts:
            logger.info("[search policy] drafting new node (not enough drafts)")
            return None

        # debugging
        if random.random() < search_cfg.debug_prob:
            # nodes that are buggy + leaf nodes + debug depth < max debug depth
            debuggable_nodes = [
                n
                for n in self.journal.buggy_nodes
                if (n.is_leaf and n.debug_depth <= search_cfg.max_debug_depth)
            ]
            if debuggable_nodes:
                node_to_debug = random.choice(debuggable_nodes)
                logger.info(f"[search policy] debugging node {node_to_debug.id}")
                return node_to_debug

        # back to drafting if no nodes to improve
        good_nodes = self.journal.good_nodes
        if not good_nodes:
            logger.info("[search policy] drafting new node (no good nodes)")
            return None

        # greedy
        greedy_node = self.journal.get_best_node()
        logger.info(f"[search policy] greedy node selected: node {greedy_node.id}")
        return greedy_node

    @property
    def _prompt_environment(self):
        pkgs = [
            "numpy",
            "pandas",
            "scikit-learn",
            "statsmodels",
            "xgboost",
            "lightGBM",
            "torch",
            "torchvision",
            "torch-geometric",
            "bayesian-optimization",
            "timm",
        ]
        random.shuffle(pkgs)
        pkg_str = ", ".join([f"`{p}`" for p in pkgs])

        env_prompt = {
            "Installed Packages": f"Your solution can use any relevant machine learning packages such as: {pkg_str}. Feel free to use any other packages too (all packages are already installed!). For neural networks we suggest using PyTorch rather than TensorFlow."
        }
        return env_prompt

    @property
    def _prompt_impl_guideline(self):
        tot_time_elapsed = time.time() - self.start_time
        tot_time_remaining = self.acfg.time_limit - tot_time_elapsed
        exec_timeout = int(min(self.cfg.exec.timeout, tot_time_remaining))

        impl_guideline = [
            f"<TOTAL_TIME_REMAINING: {format_time(tot_time_remaining)}>",
            f"<TOTAL_STEPS_REMAINING: {self.acfg.steps - self.current_step}>",
            "The code should **implement the proposed solution**, **print the value of the evaluation metric computed on a hold-out validation set**,",
            "**AND MOST IMPORTANTLY SAVE PREDICTIONS ON THE PROVIDED UNLABELED TEST DATA IN A `submission.csv` FILE IN THE ./submission/ DIRECTORY.**",
            "The code should be a single-file python program that is self-contained and can be executed as-is.",
            "No parts of the code should be skipped, don't terminate the before finishing the script.",
            "Your response should only contain a single code block.",
            f"Be aware of the running time of the code, it should complete within {humanize.naturaldelta(exec_timeout)}.",
            'All the provided input data is stored in "./input" directory.',
            '**You MUST submit predictions on the provided unlabeled test data in a `submission.csv` file** file in the "./working" directory as described in the task description** This is extremely important since this file is used for grading/evaluation. DO NOT FORGET THE submission.csv file!',
            'You can also use the "./working" directory to store any temporary files that your code needs to create.',
            "REMEMBER THE ./submission/submission.csv FILE!!!!! The correct directory is important too.",
        ]
        # Multi-target competitions using mean of some metric across targets
        impl_guideline.append(
            "If the competition's evaluation is the mean of the column metric across multiple target columns, compute metric per target column and average them to obtain a single mean of the metrics."
        )
        impl_guideline.append(
            "Use this mean of the metrics for validation, model selection, early stopping, and hyperparameter tuning; also print it clearly as mean_X, X being the column metric, (e.g., 'mean_rmse: <value>')."
        )
        impl_guideline.append(
            "When using k-fold CV, compute mean of the column metric within each fold, then average across folds to produce the final CV score that you print."
        )
        if self.acfg.expose_prediction:
            impl_guideline.append(
                "The implementation should include a predict() function, "
                "allowing users to seamlessly reuse the code to make predictions on new data. "
                "The prediction function should be well-documented, especially the function signature."
            )

        if self.acfg.k_fold_validation > 1:
            impl_guideline.append(
                f"The evaluation should be based on {self.acfg.k_fold_validation}-fold cross-validation but only if that's an appropriate evaluation for the task at hand."
            )

        return {"Implementation guideline": impl_guideline}

    @property
    def _prompt_resp_fmt(self):
        return {
            "Response format": (
                "Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), "
                "followed by a single markdown code block (wrapped in ```) which implements this solution and prints out the evaluation metric. "
                "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
                "If the competition evaluates mean of the some metric across multiple target columns, the code must compute the mean X averaged across targets and print it clearly labeled as 'mean_X' (mean_RMSE as an example.). "
            )
        }

    # -------- Knowledge Base Guidance Utilities --------
    def _task_text(self) -> str:
        """Return the task description as plain text for keyword detection."""
        desc = self.task_desc
        try:
            if isinstance(desc, dict):
                return compile_prompt_to_md(desc)
        except Exception:
            pass
        return str(desc)

    def _detect_problem_type(self) -> str:
        """Heuristically detect problem type to select a relevant KB example."""
        text = (self._task_text() + "\n" + (self.data_preview or "")).lower()
        cats = {
            "vision": ["image", "images", "cv", "segmentation", "detection", "object", "yolo", "vit", "cnn", "mask", "super-resolution"],
            "nlp": ["text", "nlp", "language", "bert", "gpt", "token", "sequence", "summarization", "translation", "ner", "sentiment", "qa", "question answering"],
            "timeseries": ["time series", "timeseries", "forecast", "temporal", "lag", "rolling", "autocorrelation", "trend", "seasonal", "signal"],
            "structured_data": ["tabular", "structured", "csv", "xgboost", "catboost", "random forest", "lightgbm", "feature", "columns", "row", "regression"],
            "graph": ["graph", "gnn", "node", "edge", "network", "citation"],
            "audio": ["audio", "speech", "asr", "mel", "spectrogram", "voice", "sound"],
            "rl": ["reinforcement", "policy", "environment", "agent", "cartpole", "q-learning", "ppo", "ddpg"],
            "generative": ["generate", "generation", "gan", "diffusion", "vae", "style", "dreambooth", "gpt2", "text generation"],
        }
        best_cat, best_score = "structured_data", -1
        for cat, kws in cats.items():
            score = sum(1 for kw in kws if kw in text)
            if score > best_score:
                best_cat, best_score = cat, score
        logger.info(f"[kb] Detected problem type: {best_cat} (score={best_score})")
        return best_cat

    def _select_relevant_example(self, category: str) -> tuple[str, str]:
        """
        Pick the most relevant example file path and a code snippet from the KB.
        Returns (relative_path_for_prompt, code_snippet). Empty strings if not found.
        """
        kb_dir = self.kb_root / category
        if not kb_dir.exists():
            return "", ""
        files = sorted([p for p in kb_dir.glob("*.py") if p.is_file()])
        if not files:
            return "", ""

        text = self._task_text().lower()
        # Tokenize task text to keywords for fuzzy matching
        tokens = set(re.findall(r"[a-zA-Z_]{3,}", text))

        def score_file(p: Path) -> int:
            name = p.stem.lower()
            score = 0
            # filename token matches
            for t in tokens:
                if t in name:
                    score += 1
            # small bonus for very common task patterns
            for bonus_kw in ["classification", "regression", "segmentation", "detection", "translation", "summarization", "forecast", "recommend"]:
                if bonus_kw in name:
                    score += 2
            return score

        scored = sorted(((score_file(p), p) for p in files), key=lambda x: (-x[0], x[1].name))
        best_score, best_path = scored[0]
        if best_score == 0:
            best_path = files[0]  # fallback deterministic choice

        try:
            snippet = best_path.read_text(encoding="utf-8", errors="ignore")
            # keep knowledge-base example concise to reduce prompt length while preserving structure
            snippet = trim_long_string(snippet, threshold=6000, k=3000)
        except Exception as e:
            logger.warning(f"[kb] Failed reading example {best_path}: {e}")
            return "", ""

        # Make path relative to repository root (parent of 'aide')
        try:
            repo_root = Path(__file__).parent.parent.resolve()
            rel_path = str(best_path.resolve().relative_to(repo_root))
        except Exception:
            rel_path = str(best_path)

        return rel_path, snippet

    # -------- Two-Stage Generation: Stage 1 (Plan) --------
    def generate_stage1_summary(self) -> str:
        """Generate and cache a concise, actionable plan that fits the dataset and competition context."""
        if self.stage1_summary:
            return self.stage1_summary

        # compute an execution-time hint to bound complexity in Stage 1 planning
        tot_time_elapsed = time.time() - self.start_time
        tot_time_remaining = self.acfg.time_limit - tot_time_elapsed
        exec_timeout = int(min(self.cfg.exec.timeout, tot_time_remaining))

        introduction = (
            "You are a Kaggle grandmaster. Produce a concise, actionable plan for a single-file solution "
            "tailored to the dataset and evaluation. Do not include any code."
        )
        if self.acfg.obfuscate:
            introduction = (
                "You are an expert ML engineer. Produce a concise, actionable plan for a single-file solution "
                "tailored to the dataset and evaluation. Do not include any code."
            )

        prompt: Any = {
            "Introduction": introduction,
            "Task description": self.task_desc,
            "Instructions": {
                "Output format": (
                    "Return 5–8 bullet points (no code blocks). Each bullet must be specific and actionable."
                ),
                "Must include": [
                    "Task framing (regression/classification/etc.) and target(s).",
                    "Expected input schema assumptions relevant to the dataset.",
                    "Primary model family you will use.",
                    "Validation strategy (e.g., holdout, K-fold) and the key validation metric.",
                    "Any essential preprocessing/feature engineering steps.",
                    "How to generate predictions for the test set and save to ./submission/submission.csv.",
                    f"Be mindful of runtime; the script should finish within {humanize.naturaldelta(exec_timeout)}.",
                ],
                "Prohibitions": [
                    "Do not include any code.",
                    "Keep it compact and practical, avoid long explanations.",
                ],
            },
        }
        if self.acfg.data_preview and self.data_preview:
            prompt["Data Overview"] = self.data_preview

        summary_text = query(
            system_message=prompt,
            user_message=None,
            model=self.acfg.code.model,
            temperature=max(0.1, self.acfg.code.temp * 0.7),
            convert_system_to_user=self.acfg.convert_system_to_user,
        )
        self.stage1_summary = str(summary_text).strip()
        logger.info("[stage1] Generated and cached plan summary")
        return self.stage1_summary

    def plan_and_code_query(self, prompt, retries=3) -> tuple[str, str]:
        """Generate a natural language plan + code in the same LLM call and split them apart."""
        completion_text = None
        for _ in range(retries):
            completion_text = query(
                system_message=prompt,
                user_message=None,
                model=self.acfg.code.model,
                temperature=self.acfg.code.temp,
                convert_system_to_user=self.acfg.convert_system_to_user,
            )

            code = extract_code(completion_text)
            nl_text = extract_text_up_to_code(completion_text)

            if code and nl_text:
                # merge all code blocks into a single string
                return nl_text, code

            logger.info("Plan + code extraction failed, retrying...")
        logger.info("Final plan + code extraction attempt failed, giving up...")
        return "", completion_text  # type: ignore

    def _draft(self) -> Node:
        """
        Two-stage drafting:
          - Stage 1: Generate a compact, cached plan summary tailored to the dataset and task.
          - Stage 2: Generate code using that plan. For baseline drafts, enforce a minimal baseline style.
                     For exactly one knowledge-based draft, fully leverage the KB example to produce a more advanced solution.
        """
        # Ensure Stage 1 summary exists and is cached
        stage1_plan = self.generate_stage1_summary()

        # Determine whether this draft should be the knowledge-based one
        guided_example_path: str | None = None
        ex_snippet: str | None = None
        category: str | None = None
        try:
            draft_idx = len(self.journal.draft_nodes)
            already_guided = any(
                (n.parent is None) and (getattr(n, "guided_example_path", None) is not None)
                for n in self.journal.nodes
            )
            should_guide = (
                draft_idx < self.acfg.search.num_drafts
                and not already_guided
                and draft_idx == self.guided_draft_index
            )
            if should_guide:
                category = self._detect_problem_type()
                ex_path, kb_snippet = self._select_relevant_example(category)
                if ex_path and kb_snippet:
                    guided_example_path = ex_path
                    ex_snippet = kb_snippet
                    logger.info(f"[draft] Knowledge-based draft selected (idx={draft_idx}) with KB example {ex_path}")
        except Exception as e:
            logger.warning(f"[draft] KB selection failed: {e}")

        # Build Stage 2 prompt
        base_intro = (
            "You are a Kaggle grandmaster attending a competition. "
            "Based on the Stage 1 plan below, implement the solution in a single Python file."
        )
        kb_intro = (
            "You are a Kaggle grandmaster attending a competition. "
            "Based on the Stage 1 plan and the knowledge-base example below, implement a highly optimized, "
            "context-aware solution that fully leverages the knowledge base."
        )
        if self.acfg.obfuscate:
            base_intro = (
                "You are an expert machine learning engineer. "
                "Based on the Stage 1 plan below, implement the solution in a single Python file."
            )
            kb_intro = (
                "You are an expert machine learning engineer. "
                "Based on the Stage 1 plan and the knowledge-base example below, implement a highly optimized, "
                "context-aware solution that fully leverages the knowledge base."
            )

        prompt: Any = {
            "Introduction": kb_intro if ex_snippet else base_intro,
            "Task description": self.task_desc,
            "Stage 1 plan": stage1_plan,
            "Instructions": {},
        }

        # Response format + critical implementation requirements
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= self._prompt_impl_guideline

        # Draft-style specific guidance
        if ex_snippet:
            prompt["Instructions"] |= {
                "Knowledge-based drafting guideline": [
                    "Leverage the structure, modeling choices, and patterns from the knowledge base example.",
                    "Adapt to the present dataset and evaluation; do not copy irrelevant pieces.",
                    "It is acceptable to use stronger models, advanced feature engineering, CV, early stopping, "
                    "and reasonable optimization that fits within the runtime budget.",
                ]
            }
            prompt["Knowledge base guidance"] = [
                f"A relevant '{category}' example from our knowledge base is provided below.",
                "Borrow ideas, patterns, and structure as appropriate; adapt to this dataset and metric.",
            ]
            prompt["Knowledge base example"] = {
                "Snippet": wrap_code(ex_snippet),
            }
        else:
            prompt["Instructions"] |= {
                "Baseline drafting guideline": [
                    "Produce a simple, minimal baseline-style implementation.",
                    "Prefer straightforward models and defaults; avoid ensembling and heavy hyper-parameter tuning.",
                    "Keep preprocessing light and pragmatic; prioritize reliability and clarity.",
                ]
            }

        # Keep prompts compact in Stage 2 (omit Data Overview and Environment here);
        # the Stage 1 plan already captures the essential context.

        plan, code = self.plan_and_code_query(prompt)
        new_node = Node(plan=plan, code=code, guided_example_path=guided_example_path)
        logger.info(f"Drafted new node {new_node.id}")
        return new_node

    def _improve(self, parent_node: Node) -> Node:
        introduction = (
            "You are a Kaggle grandmaster attending a competition. You are provided with a previously developed "
            "solution below and should improve it in order to further increase the (test time) performance. "
            "For this you should first outline a brief plan in natural language for how the solution can be improved and "
            "then implement this improvement in Python based on the provided previous solution. "
        )
        if self.acfg.obfuscate:
            introduction = (
                "You are an expert machine learning engineer attempting a task. You are provided with a previously developed "
                "solution below and should improve it in order to further increase the (test time) performance. "
                "For this you should first outline a brief plan in natural language for how the solution can be improved and "
                "then implement this improvement in Python based on the provided previous solution. "
            )
        # Reference Stage 1 plan to keep improvement prompts compact and consistent
        stage1_plan = self.generate_stage1_summary()
        prompt: Any = {
            "Introduction": introduction,
            "Task description": self.task_desc,
            "Stage 1 plan": stage1_plan,
            "Memory": self.journal.generate_summary(),
            "Instructions": {},
        }
        prompt["Previous solution"] = {
            "Code": wrap_code(parent_node.code),
        }

        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Solution improvement sketch guideline": [
                "The solution sketch should be a brief natural language description of how the previous solution can be improved.",
                "You should be very specific and should only propose a single actionable improvement.",
                "This improvement should be atomic so that we can experimentally evaluate the effect of the proposed change.",
                "Take the Memory section into consideration when proposing the improvement.",
                "The solution sketch should be 3-5 sentences.",
                "Don't suggest to do EDA.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline

        plan, code = self.plan_and_code_query(prompt)
        new_node = Node(plan=plan, code=code, parent=parent_node)
        logger.info(f"Improved node {parent_node.id} to create new node {new_node.id}")
        return new_node

    def _debug(self, parent_node: Node) -> Node:
        introduction = (
            "You are a Kaggle grandmaster attending a competition. "
            "Your previous solution had a bug and/or did not produce a submission.csv, "
            "so based on the information below, you should revise it in order to fix this. "
            "Your response should be an implementation outline in natural language,"
            " followed by a single markdown code block which implements the bugfix/solution."
        )
        if self.acfg.obfuscate:
            introduction = (
                "You are an expert machine learning engineer attempting a task. "
                "Your previous solution had a bug and/or did not produce a submission.csv, "
                "so based on the information below, you should revise it in order to fix this. "
                "Your response should be an implementation outline in natural language,"
                " followed by a single markdown code block which implements the bugfix/solution."
            )
        # Reference Stage 1 plan to keep debug prompts compact and anchored
        stage1_plan = self.generate_stage1_summary()
        prompt: Any = {
            "Introduction": introduction,
            "Task description": self.task_desc,
            "Stage 1 plan": stage1_plan,
            "Previous (buggy) implementation": wrap_code(parent_node.code),
            "Execution output": wrap_code(parent_node.term_out, lang=""),
            "Instructions": {},
        }
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Bugfix improvement sketch guideline": [
                "You should write a brief natural language description (3-5 sentences) of how the issue in the previous implementation can be fixed.",
                "Don't suggest to do EDA.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline

        if self.acfg.data_preview:
            prompt["Data Overview"] = self.data_preview

        plan, code = self.plan_and_code_query(prompt)
        new_node = Node(plan=plan, code=code, parent=parent_node)
        logger.info(f"Debugged node {parent_node.id} to create new node {new_node.id}")
        return new_node

    def update_data_preview(
        self,
    ):
        # Prefer compact EDA Markdown if available
        try:
            eda_compact = self.cfg.workspace_dir / "EDA_COMPACT.md"
            if eda_compact.exists():
                self.data_preview = eda_compact.read_text()
                return
        except Exception:
            pass
        # Fallback to legacy directory preview
        self.data_preview = data_preview.generate(self.cfg.workspace_dir)

    def step(self, exec_callback: ExecCallbackType):
        # clear the submission dir from previous steps
        shutil.rmtree(self.cfg.workspace_dir / "submission", ignore_errors=True)
        (self.cfg.workspace_dir / "submission").mkdir(exist_ok=True)

        if not self.journal.nodes or self.data_preview is None:
            self.update_data_preview()

        parent_node = self.search_policy()
        logger.info(f"Agent is generating code, parent node type: {type(parent_node)}")

        if parent_node is None:
            result_node = self._draft()
        elif parent_node.is_buggy:
            result_node = self._debug(parent_node)
        else:
            result_node = self._improve(parent_node)

        result_node = self.parse_exec_result(
            node=result_node,
            exec_result=exec_callback(result_node.code, True),
        )
        # handle final cases where we missed buggy nodes somehow
        if not result_node.is_buggy:
            if not (self.cfg.workspace_dir / "submission" / "submission.csv").exists():
                result_node.is_buggy = True
                result_node.metric = WorstMetricValue()
                logger.info(
                    f"Actually, node {result_node.id} did not produce a submission.csv"
                )
        self.journal.append(result_node)

        # if the result_node is the best node, cache its submission.csv and solution.py
        # to best_solution/ by copying it there
        best_node = self.journal.get_best_node()
        if best_node is not None:
            if best_node.id == result_node.id:
                logger.info(f"Node {result_node.id} is the best node so far")
                best_solution_dir = self.cfg.workspace_dir / "best_solution"
                best_solution_dir.mkdir(exist_ok=True, parents=True)
                # copy submission/submission.csv to best_submission/submission.csv
                best_submission_dir = self.cfg.workspace_dir / "best_submission"
                best_submission_dir.mkdir(exist_ok=True, parents=True)
                shutil.copy(
                    self.cfg.workspace_dir / "submission" / "submission.csv",
                    best_submission_dir,
                )
                # copy solution.py and relevant node id to best_solution/
                with open(best_solution_dir / "solution.py", "w") as f:
                    f.write(result_node.code)
                # take note of the node id of the best node
                with open(best_solution_dir / "node_id.txt", "w") as f:
                    f.write(str(result_node.id))
            else:
                logger.info(f"Node {result_node.id} is not the best node")
                logger.info(f"Node {best_node.id} is still the best node")
        self.current_step += 1

    def parse_exec_result(self, node: Node, exec_result: ExecutionResult) -> Node:
        logger.info(f"Agent is parsing execution results for node {node.id}")

        node.absorb_exec_result(exec_result)

        introduction = (
            "You are a Kaggle grandmaster attending a competition. "
            "You have written code to solve this task and now need to evaluate the output of the code execution. "
            "You should determine if there were any bugs as well as report the empirical findings. "
            "If the competition uses mean of some metric across multiple target columns, base your evaluation on the mean of the metric averaged across the target columns. "
            "If multiple metrics are printed, prefer a metric clearly labeled 'mean_X' or the single overall validation score that averages X across targets, X being the metric like RMSE."
        )
        if self.acfg.obfuscate:
            introduction = (
                "You are an expert machine learning engineer attempting a task. "
                "You have written code to solve this task and now need to evaluate the output of the code execution. "
                "You should determine if there were any bugs as well as report the empirical findings. "
                "If the task uses mean of the some metric across multiple target columns, base your evaluation on the mean of the some metric averaged across the target columns. "
                "If multiple metrics are printed, prefer a metric clearly labeled 'mean_X' or the single overall validation score that averages X across targets, X being the metric like RMSE."
            )
        prompt = {
            "Introduction": introduction,
            "Task description": self.task_desc,
            "Implementation": wrap_code(node.code),
            "Execution output": wrap_code(node.term_out, lang=""),
        }

        response = cast(
            dict,
            query(
                system_message=prompt,
                user_message=None,
                func_spec=review_func_spec,
                model=self.acfg.feedback.model,
                temperature=self.acfg.feedback.temp,
                convert_system_to_user=self.acfg.convert_system_to_user,
            ),
        )

        # if the metric isn't a float then fill the metric with the worst metric
        if not isinstance(response["metric"], float):
            response["metric"] = None

        # do an extra check, to catch cases where judge fails
        has_csv_submission = (
            self.cfg.workspace_dir / "submission" / "submission.csv"
        ).exists()

        node.analysis = response["summary"]
        node.is_buggy = (
            response["is_bug"]
            or node.exc_type is not None
            or response["metric"] is None
            or response["has_csv_submission"] == False
            or has_csv_submission == False
        )

        if node.is_buggy:
            logger.info(
                f"Parsed results: Node {node.id} is buggy and/or did not produce a submission.csv"
            )
            node.metric = WorstMetricValue()
        else:
            logger.info(f"Parsed results: Node {node.id} is not buggy")
            node.metric = MetricValue(
                response["metric"], maximize=not response["lower_is_better"]
            )

        return node
