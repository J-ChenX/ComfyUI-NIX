import { app } from "/scripts/app.js"; // 如有 404，请改为 "../../scripts/app.js" 或 "../../../scripts/app.js"

app.registerExtension({
  name: "nix.switch-anything.dynamic-inputs",

  init() {
    // 可选：在粘贴期间“冻结”，避免过程中被提前裁剪
    const original_paste = LGraphCanvas.prototype.pasteFromClipboard;
    if (original_paste && !LGraphCanvas.prototype.__nixPatchedPaste) {
      LGraphCanvas.prototype.__nixPatchedPaste = true;
      LGraphCanvas.prototype.pasteFromClipboard = function () {
        try {
          window.__NIX_PASTING__ = true;
          return original_paste.apply(this, arguments);
        } finally {
          window.__NIX_PASTING__ = false;
        }
      };
    }
  },

  beforeRegisterNodeDef(nodeType, nodeData, app) {
    const className = nodeData?.name || nodeType?.comfyClass || nodeType?.type;
    if (className !== "NIX_SwitchAnything") return;

    const MAX_INPUTS = 32; // 与后端保持一致

    // 不要在创建时变更输入口，避免克隆/反序列化早期阶段被破坏
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function (...args) {
      return onNodeCreated?.apply(this, args);
    };

    // 在 configure 完成后，安排一个“延迟收敛”，以便对“没有任何连接恢复”的复制场景也能生效
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      const r = onConfigure?.apply(this, arguments);
      scheduleNormalize(this, 250); // 初次安排，若后续有连接变更会被重置
      return r;
    };

    // 连接变更：先让原逻辑执行，再根据 connect 做“补空/安排延迟收敛”
    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (side, slot, connect, link_info, output) {
      const r = onConnectionsChange?.apply(this, arguments);

      const INPUT_SIDE = (window.LiteGraph && LiteGraph.INPUT !== undefined) ? LiteGraph.INPUT : 1;
      if (side === INPUT_SIDE) {
        // 正在建立连接：只“补空”不“裁剪”，以免破坏后续恢复
        if (connect) {
          const hasFree = getAnythingInputInfos(this).some(info => !info.inp.link);
          if (!hasFree && highestPresentIndex(this) < MAX_INPUTS) {
            addNextAnythingInput(this);
          }
        }
        // 无论建立还是断开，安排一次“抖动延迟收敛”
        scheduleNormalize(this, 120);
        this.setDirtyCanvas?.(true, true);
      }

      return r;
    };

    // ============ 工具函数 ============

    function scheduleNormalize(node, delayMs) {
      if (!node) return;
      // 粘贴过程中先跳过（Ctrl+V），结束后由后续变更触发的计时器来收敛
      if (window.__NIX_PASTING__) return;
      clearTimeout(node.__nixNormalizeTimer);
      node.__nixNormalizeTimer = setTimeout(() => {
        try {
          normalizeAnythingInputs(node);
        } catch (e) {
          console.error(e);
        }
      }, delayMs);
    }

    function getAnythingInputInfos(node) {
      const ret = [];
      (node.inputs || []).forEach((inp, idx) => {
        const m = inp?.name?.match?.(/^anything_(\d+)$/);
        if (m) ret.push({ idx, inp, n: parseInt(m[1], 10) });
      });
      ret.sort((a, b) => a.n - b.n);
      return ret;
    }

    function highestPresentIndex(node) {
      const infos = getAnythingInputInfos(node);
      if (infos.length === 0) return 1; // 正常应至少有 anything_1
      return infos[infos.length - 1].n;
    }

    function addNextAnythingInput(node) {
      const next = Math.min(MAX_INPUTS, Math.max(2, highestPresentIndex(node) + 1));
      const name = `anything_${next}`;
      const exists = (node.inputs || []).some(i => i?.name === name);
      if (!exists) {
        node.addInput(name, "*"); // 前端通配类型，对应后端 IO.ANY
      }
    }

    // 延迟收敛：保留所有“已连接”的 anything_*；在“未连接”的 anything_* 中仅保留 1 个（若 0 个则新增 1 个）
    function normalizeAnythingInputs(node) {
      if (!node?.inputs) return;

      const infos = getAnythingInputInfos(node);
      if (infos.length === 0) {
        // 极端兜底（后端没给 anything_1）
        node.addInput("anything_2", "*");
        node.setDirtyCanvas?.(true, true);
        return;
      }

      // 按“是否已连接”分组
      const connected = [];
      const free = [];
      infos.forEach(info => {
        if (info.inp.link) connected.push(info);
        else free.push(info);
      });

      // 确保“未连接”的只保留 1 个，删除多余的（按 idx 降序删，避免位移）
      if (free.length === 0) {
        if (highestPresentIndex(node) < MAX_INPUTS) {
          addNextAnythingInput(node);
        }
      } else if (free.length > 1) {
        const toRemove = free.slice(1).sort((a, b) => b.idx - a.idx);
        toRemove.forEach(info => node.removeInput(info.idx));
      }

      node.setDirtyCanvas?.(true, true);
    }
  },
});
