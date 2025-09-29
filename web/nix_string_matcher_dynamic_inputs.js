import { app } from "../../scripts/app.js";

app.registerExtension({
  name: "nix.string-matcher.dynamic-pairs.final",

  beforeRegisterNodeDef(nodeType, nodeData, app) {
    const className = nodeData?.name || nodeType?.comfyClass || nodeType?.type;
    if (className !== "NIX_StringMatcher") return;

    const MAX_N = 32;
    const MIN_N = 2;

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function (...args) {
      const r = origOnNodeCreated?.apply(this, args);
      ensureUpdateButton(this);
      ensureJudgeCombo(this);         // 新增：把 judge_string 变为下拉
      snapshotDefaults(this);

      // 新建节点：仅保留 2 个（string_1、string_2）
      applyTargetCount(this, 2, { keepLinked: false });

      updateJudgeChoices(this);       // 新增：按当前 string 值填充选项
      reorderWidgets(this);
      safeRefresh(this);
      return r;
    };

    const origOnConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      const r = origOnConfigure?.apply(this, arguments);
      ensureUpdateButton(this);
      ensureJudgeCombo(this);         // 新增：恢复时确保是下拉
      snapshotDefaults(this);

      // 恢复工程时：以 inputcount 为准，但保留已连线的输入
      let target = getInputCount(this) ?? 2;
      target = clamp(target, MIN_N, MAX_N);
      applyTargetCount(this, target, { keepLinked: true });

      updateJudgeChoices(this);       // 新增：按当前 string 值填充选项
      reorderWidgets(this);
      safeRefresh(this);
      return r;
    };

    // 以 inputcount 为基准，动态增删 string_3..string_target，删除 >target 的（可选保留已连线）
    function applyTargetCount(node, target, { keepLinked }) {
      target = clamp(target, MIN_N, MAX_N);

      if (keepLinked) {
        const maxLinked = getMaxLinkedIndex(node);
        if (maxLinked > target) target = maxLinked;
        setInputCount(node, target);
      }

      // 删除多余的（从大编号向下，避免索引错位）
      for (let i = MAX_N; i >= 3; i--) {
        if (i > target) {
          if (keepLinked && isLinked(node, i)) continue;
          removePair(node, i);
        }
      }

      // 补齐不足的（3..target）
      for (let i = 3; i <= target; i++) {
        ensurePair(node, i);
      }

      // 自愈：确保 3..target 范围内，每个编号的“端口+小部件”都齐全并互相关联
      for (let i = 3; i <= target; i++) {
        const name = `string_${i}`;
        if (!hasInput(node, name) || !hasWidget(node, name)) {
          ensurePair(node, i);
        } else {
          const idx = findInputIndex(node, name);
          const w = getWidget(node, name);
          if (idx >= 0 && node.inputs[idx]) node.inputs[idx].widget = w;
          if (w) w.input = name;
        }
      }

      safeRefresh(node);
    }

    // 确保编号 i 的“端口”和“文本小部件”都存在，并进行互相关联
    function ensurePair(node, i) {
      const name = `string_${i}`;

      // 1) 确保端口存在
      if (!hasInput(node, name)) {
        try {
          node.addInput(name, "STRING");
        } catch (e) {
          console.error(`addInput failed for ${name}`, e);
        }
      }

      // 2) 确保文本小部件存在
      if (!hasWidget(node, name)) {
        try {
          const def = getDefaultValue(node, i) ?? "";
          const opts = { ...(getDefaultOptions(node, i) || {}), serialize: true };
          const w = node.addWidget("string", name, def, null, opts);
          if (w) w.input = name;
        } catch (e) {
          console.error(`addWidget failed for ${name}`, e);
        }
      }

      // 3) 互相关联
      const idx = findInputIndex(node, name);
      const w = getWidget(node, name);
      if (idx >= 0 && node.inputs[idx]) node.inputs[idx].widget = w;
      if (w) w.input = name;
    }

    // 删除编号 i 的端口与小部件（先断开连线）
    function removePair(node, i) {
      const name = `string_${i}`;

      // 删端口
      const idx = findInputIndex(node, name);
      if (idx >= 0) {
        try { node.disconnectInput(idx); } catch {}
        try { node.removeInput(idx); } catch (e) {
          console.warn(`removeInput failed for ${name}`, e);
        }
      }

      // 删小部件
      const wIdx = findWidgetIndex(node, name);
      if (wIdx >= 0) {
        try { node.widgets.splice(wIdx, 1); } catch (e) {
          console.warn(`remove widget failed for ${name}`, e);
        }
      }
    }

    // 确保按钮存在并位于 inputcount 后面
    function ensureUpdateButton(node) {
      if (!node.widgets) node.widgets = [];
      let btn = node.widgets.find(w => w.type === "button" && w.name === "Update inputs");
      if (!btn) {
        btn = node.addWidget("button", "Update inputs", null, () => {
          let target = getInputCount(node);
          if (!Number.isFinite(target)) {
            try { app.ui?.showToast?.("NIX_StringMatcher: 未找到 inputcount", "error"); } catch {}
            return;
          }
          target = clamp(Math.floor(target), MIN_N, MAX_N);
          // 点击按钮：允许删除多余项（即使有连线）
          applyTargetCount(node, target, { keepLinked: false });
          reorderWidgets(node);
          updateJudgeChoices(node);     // 新增：点击后刷新 judge 下拉选项
          safeRefresh(node);
        });
      }
      const ws = node.widgets;
      const btnIdx = ws.indexOf(btn);
      const icIdx = ws.findIndex(w => w.name === "inputcount");
      if (btnIdx >= 0) {
        ws.splice(btnIdx, 1);
        ws.splice(icIdx >= 0 ? icIdx + 1 : 0, 0, btn);
      }
    }

    // 把 judge_string 从 "string" 改造成 "combo"（保留同名以便序列化给后端）
    function ensureJudgeCombo(node) {
      if (!node.widgets) node.widgets = [];
      const idx = findWidgetIndex(node, "judge_string");
      const existed = idx >= 0 ? node.widgets[idx] : null;
      const prevVal = existed?.value ?? "";

      if (existed && existed.type === "combo") {
        // 已经是下拉了，确保可序列化
        existed.options = existed.options || {};
        if (existed.options.serialize !== true) existed.options.serialize = true;
        return;
      }

      // 移除旧的字符串输入
      if (idx >= 0) {
        try { node.widgets.splice(idx, 1); } catch {}
      }

      // 新建下拉
      try {
        const w = node.addWidget("combo", "judge_string", prevVal, null, {
          values: [],        // 稍后用 updateJudgeChoices 填充
          serialize: true
        });
        // 无需绑定 input 端口，judge_string 本身就是一个参数小部件
      } catch (e) {
        console.error("create judge_string combo failed", e);
      }
    }

    // 收集当前所有 string_i 的值作为下拉选项
    function collectStringChoices(node) {
      const target = clamp(getInputCount(node) ?? 2, MIN_N, MAX_N);
      const seen = new Set();
      const choices = [];
      for (let i = 1; i <= target; i++) {
        const w = getWidget(node, `string_${i}`);
        if (!w) continue;
        let v = w.value;
        if (v == null) continue;
        v = String(v);
        // 如需允许选择空字符串，把下面这行改为：if (!seen.has(v)) { ... }
        if (v.length === 0) continue;
        if (!seen.has(v)) {
          seen.add(v);
          choices.push(v);
        }
      }
      return choices;
    }

    // 用收集到的值更新 judge_string 的下拉选项
    function updateJudgeChoices(node) {
      const w = getWidget(node, "judge_string");
      if (!w) return;
      const cur = w.value ?? "";
      const choices = collectStringChoices(node);
      if (cur && !choices.includes(cur)) choices.unshift(cur);
      w.options = w.options || {};
      w.options.values = choices.length ? choices : [cur ?? ""];
    }

    // 按指定顺序重排 widgets：inputcount -> 按钮 -> judge_string -> string_1 -> string_2 -> string_3.. -> 其他
    function reorderWidgets(node) {
      if (!node.widgets || !node.widgets.length) return;

      const widgets = node.widgets.slice();
      const pick = n => widgets.find(w => w.name === n);
      const btn = widgets.find(w => w.type === "button" && w.name === "Update inputs");
      const wInputCount = pick("inputcount");
      const wJudge = pick("judge_string");
      const wS1 = pick("string_1");
      const wS2 = pick("string_2");

      const restStrings = widgets
        .filter(w => {
          if (!w?.name) return false;
          if (w === wInputCount || w === btn || w === wJudge || w === wS1 || w === wS2) return false;
          return /^string_(\d+)$/.test(w.name);
        })
        .sort((a, b) => {
          const na = parseInt(a.name.split("_")[1], 10);
          const nb = parseInt(b.name.split("_")[1], 10);
          return na - nb;
        });

      const others = widgets.filter(w => {
        if (!w?.name) return false;
        if (w === wInputCount || w === btn || w === wJudge || w === wS1 || w === wS2) return false;
        if (/^string_(\d+)$/.test(w.name)) return false;
        return true;
      });

      const ordered = [];
      if (wInputCount) ordered.push(wInputCount);
      if (btn) ordered.push(btn);
      if (wJudge) ordered.push(wJudge);
      if (wS1) ordered.push(wS1);
      if (wS2) ordered.push(wS2);
      ordered.push(...restStrings, ...others);

      node.widgets.length = 0;
      ordered.forEach(w => node.widgets.push(w));
    }

    // 记录初始默认值/选项，便于重新添加时还原
    function snapshotDefaults(node) {
      if (node.__nix_defaults) return;
      node.__nix_defaults = { values: {}, options: {} };
      for (let i = 3; i <= MAX_N; i++) {
        const name = `string_${i}`;
        const w = node.widgets?.find(w => w.name === name);
        if (w) {
          node.__nix_defaults.values[i] = w.value;
          node.__nix_defaults.options[i] = { ...(w.options || {}) };
        }
      }
    }

    // 辅助函数
    function getDefaultValue(node, i) {
      return node.__nix_defaults?.values?.[i];
    }
    function getDefaultOptions(node, i) {
      return node.__nix_defaults?.options?.[i];
    }

    function getInputCount(node) {
      const w = node.widgets?.find(w => w.name === "inputcount");
      let v = w?.value;
      if (typeof v !== "number") v = parseInt(v, 10);
      return Number.isFinite(v) ? v : null;
    }
    function setInputCount(node, v) {
      const w = node.widgets?.find(w => w.name === "inputcount");
      if (w) w.value = clamp(Math.floor(v), MIN_N, MAX_N);
    }

    function hasInput(node, name) {
      return (node.inputs || []).some(i => i?.name === name);
    }
    function findInputIndex(node, name) {
      const arr = node.inputs || [];
      for (let i = 0; i < arr.length; i++) if (arr[i]?.name === name) return i;
      return -1;
    }
    function hasWidget(node, name) {
      return (node.widgets || []).some(w => w?.name === name);
    }
    function getWidget(node, name) {
      return node.widgets?.find(w => w?.name === name);
    }
    function findWidgetIndex(node, name) {
      const arr = node.widgets || [];
      for (let i = 0; i < arr.length; i++) if (arr[i]?.name === name) return i;
      return -1;
    }
    function isLinked(node, i) {
      const idx = findInputIndex(node, `string_${i}`);
      return idx >= 0 && node.inputs[idx]?.link != null;
    }
    function getMaxLinkedIndex(node) {
      let maxN = 0;
      (node.inputs || []).forEach(inp => {
        const nm = inp?.name;
        if (!nm || !nm.startsWith("string_")) return;
        const n = parseInt(nm.split("_")[1], 10);
        if (Number.isFinite(n) && inp.link != null) maxN = Math.max(maxN, n);
      });
      return maxN;
    }
    function clamp(v, a, b) {
      return Math.max(a, Math.min(b, v));
    }
    function safeRefresh(node) {
      try { node.setDirtyCanvas(true, true); } catch {}
      try {
        const sz = node.computeSize?.(node.size) || node.size;
        if (typeof node.onResize === "function") node.onResize(sz);
      } catch {}
    }
  },
});
