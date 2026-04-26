import io
import os
import re
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse, parse_qs, urlencode
from natsort import natsorted
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_ate_ir(folder_name):
    m_push = re.search(r'AtE(-?\d+\.\d+)', folder_name)
    m_pull = re.search(r'Ir(-?\d+\.\d+)', folder_name)
    push = float(m_push.group(1)) if m_push else None
    pull = float(m_pull.group(1)) if m_pull else None
    return push, pull


def _find_exp_segment_idx(path_parts):
    fallback = None
    for i, seg in enumerate(path_parts):
        has_pp = bool(re.search(r'AtE|Ir', seg))
        if not has_pp:
            continue
        if fallback is None:
            fallback = i
        if 'esd' in seg:
            return i
    return fallback


def _rewrite_path(display_path, push, pull):
    parts = display_path.split('/')
    idx = _find_exp_segment_idx(parts)
    if idx is None:
        return display_path
    seg = parts[idx]
    if push is not None:
        seg = re.sub(r'AtE-?\d+\.\d+', f'AtE{push:.2f}', seg)
    if pull is not None:
        seg = re.sub(r'Ir-?\d+\.\d+', f'Ir{pull:.2f}', seg)
    parts[idx] = seg
    return '/'.join(parts)


def _parse_float_list(raw):
    if not raw:
        return []
    tokens = re.split(r'[,\s]+', raw.strip())
    result = []
    for t in tokens:
        try:
            result.append(float(t))
        except ValueError:
            pass
    return result


def _parse_str_list(raw):
    if not raw:
        return []
    return [t.strip() for t in raw.split(',') if t.strip()]


def _parse_token_selection(exp_segment):
    m = re.search(r'G[\d.]+-([\w-]+)\.rs', exp_segment)
    return m.group(1) if m else None


def _rewrite_token_selection(display_path, token_sel):
    parts = display_path.split('/')
    idx = _find_exp_segment_idx(parts)
    if idx is None:
        return display_path
    parts[idx] = re.sub(r'(G[\d.]+-)([\w-]+)(\.rs)', lambda m: m.group(1) + token_sel + m.group(3), parts[idx])
    return '/'.join(parts)


def _get_image_files(abs_dir):
    try:
        files = os.listdir(abs_dir)
    except OSError:
        return []
    return natsorted(
        f for f in files
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.gif'))
    )


def _build_pretrained_rel_dir(display_path, base_cwd):
    parts = [p for p in display_path.split('/') if p]
    exp_idx = _find_exp_segment_idx(parts)
    if exp_idx is None:
        return None
    skip_until = exp_idx + 1
    while skip_until < len(parts) and re.match(r'^step\d+$', parts[skip_until], re.IGNORECASE):
        skip_until += 1
    prefix_parts = parts[:exp_idx]
    suffix_parts = parts[skip_until:]
    pretrained_parts = prefix_parts + ['original_pretrained_sd1.4_bf16'] + suffix_parts
    return '/'.join(pretrained_parts)


def _parse_unlearned_concept(exp_segment):
    m = re.search(r'_U\.([^_/]+)', exp_segment)
    if not m:
        return None
    return m.group(1)


def _rewrite_concept(display_path, concept):
    parts = display_path.split('/')
    idx = _find_exp_segment_idx(parts)
    if idx is None:
        return display_path
    parts[idx] = re.sub(r'(_U\.)([^_/]+)', lambda m: m.group(1) + concept, parts[idx])
    return '/'.join(parts)


def _rewrite_prompt(display_path, prompt, exp_idx):
    parts = [p for p in display_path.split('/') if p]
    if len(parts) < 2:
        return display_path
    parts[-2] = prompt
    result = ('/' if display_path.startswith('/') else '') + '/'.join(parts)
    if display_path.endswith('/'):
        result += '/'
    return result


# ---------------------------------------------------------------------------
# Fuzzy folder picker JS + CSS (injected into every page)
# ---------------------------------------------------------------------------

FUZZY_PICKER_JS = r"""
<style>
#fp-overlay {
  display: none; position: fixed; inset: 0; z-index: 9999;
  background: rgba(0,0,0,0.6); align-items: flex-start; justify-content: center;
  padding-top: 12vh;
}
#fp-overlay.open { display: flex; }
#fp-box {
  background: #1a1a1a; border: 1px solid #444; border-radius: 8px;
  width: min(680px, 90vw); box-shadow: 0 8px 40px rgba(0,0,0,0.7);
  display: flex; flex-direction: column; overflow: hidden;
}
#fp-input {
  width: 100%; box-sizing: border-box;
  background: #111; border: none; border-bottom: 1px solid #333;
  color: #eee; font-size: 15px; padding: 14px 16px; outline: none;
  font-family: monospace;
}
#fp-list {
  max-height: 360px; overflow-y: auto; list-style: none; margin: 0; padding: 6px 0;
}
#fp-list li {
  padding: 8px 16px; font-size: 13px; font-family: monospace;
  color: #ccc; cursor: pointer; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
#fp-list li.active { background: #2a4a6a; color: #fff; }
#fp-list li:hover { background: #222; }
#fp-list li mark { background: none; color: #5fd7ff; font-weight: bold; }
#fp-hint {
  padding: 6px 16px; font-size: 11px; color: #555; border-top: 1px solid #222;
}
</style>
<div id="fp-overlay">
  <div id="fp-box">
    <input id="fp-input" placeholder="Type to filter folders..." autocomplete="off" spellcheck="false">
    <ul id="fp-list"></ul>
    <div id="fp-hint">↑↓ navigate &nbsp;&middot;&nbsp; Enter to open &nbsp;&middot;&nbsp; Esc to close</div>
  </div>
</div>
<script>
(function() {
  const overlay = document.getElementById('fp-overlay');
  const input   = document.getElementById('fp-input');
  const list    = document.getElementById('fp-list');

  let allFolders = [];
  let filtered   = [];
  let activeIdx  = 0;
  let loaded     = false;

  function fuzzyMatch(str, query) {
    if (!query) return { match: true, score: 0, html: str };
    const s = str.toLowerCase();
    const q = query.toLowerCase();
    let si = 0, qi = 0, score = 0;
    const marks = new Array(str.length).fill(false);
    let consecutive = 0;
    while (si < s.length && qi < q.length) {
      if (s[si] === q[qi]) {
        marks[si] = true;
        qi++;
        score += 1 + consecutive;
        consecutive++;
      } else {
        consecutive = 0;
      }
      si++;
    }
    if (qi < q.length) return { match: false };
    // Build highlighted HTML
    let html = '';
    for (let i = 0; i < str.length; i++) {
      const c = str[i].replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
      html += marks[i] ? `<mark>${c}</mark>` : c;
    }
    return { match: true, score, html };
  }

  function render() {
    const q = input.value;
    const results = [];
    for (const f of allFolders) {
      const r = fuzzyMatch(f, q);
      if (r.match) results.push({ name: f, score: r.score, html: r.html });
    }
    results.sort((a, b) => b.score - a.score);
    filtered = results;
    activeIdx = 0;
    list.innerHTML = '';
    results.forEach((r, i) => {
      const li = document.createElement('li');
      li.innerHTML = r.html;
      if (i === 0) li.classList.add('active');
      li.addEventListener('mouseenter', () => setActive(i));
      li.addEventListener('click', () => navigate(i));
      list.appendChild(li);
    });
  }

  function setActive(i) {
    const items = list.querySelectorAll('li');
    if (items[activeIdx]) items[activeIdx].classList.remove('active');
    activeIdx = Math.max(0, Math.min(i, items.length - 1));
    if (items[activeIdx]) {
      items[activeIdx].classList.add('active');
      items[activeIdx].scrollIntoView({ block: 'nearest' });
    }
  }

  function navigate(i) {
    if (!filtered[i]) return;
    const base = window.location.pathname.replace(/\/?$/, '/');
    window.location.href = base + encodeURIComponent(filtered[i].name) + '/';
  }

  async function open() {
    overlay.classList.add('open');
    input.value = '';
    if (!loaded) {
      const res = await fetch('/__folders__?dir=' + encodeURIComponent(window.location.pathname));
      const data = await res.json();
      allFolders = data.folders || [];
      loaded = true;
    }
    render();
    input.focus();
  }

  function close() {
    overlay.classList.remove('open');
  }

  input.addEventListener('input', render);

  input.addEventListener('keydown', e => {
    if (e.key === 'ArrowDown') { e.preventDefault(); setActive(activeIdx + 1); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setActive(activeIdx - 1); }
    else if (e.key === 'Enter') { e.preventDefault(); navigate(activeIdx); }
    else if (e.key === 'Escape') { close(); }
  });

  overlay.addEventListener('mousedown', e => { if (e.target === overlay) close(); });

  document.addEventListener('keydown', e => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'i') {
      e.preventDefault();
      loaded = false;
      if (overlay.classList.contains('open')) close(); else open();
    }
    if (e.key === 'Escape' && overlay.classList.contains('open')) close();
  });
})();
</script>
"""


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

class GalleryHTTPRequestHandler(SimpleHTTPRequestHandler):

    def _parent_href(self):
        p = self.path
        if not p.endswith('/'):
            p += '/'
        p = p.rstrip('/')
        if p == '' or p == '/':
            return None
        parent = p.rsplit('/', 1)[0]
        return parent + '/' if parent else '/'

    def do_GET(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)

        if parsed.path == "/__folders__":
            self._serve_folders(qs)
            return

        if parsed.path == "/__concat__":
            self._serve_concat_image(qs)
            return

        if parsed.path == "/__multiconcat__":
            self._serve_multiconcat_image(qs)
            return

        if parsed.path == "/__vstack__":
            self._serve_vstack_image(qs)
            return

        super().do_GET()

    # ------------------------------------------------------------------
    # /__folders__ — return JSON list of subdirectories for the picker
    # ------------------------------------------------------------------

    def _serve_folders(self, qs):
        try:
            rel_dir = qs.get("dir", ["/"])[0]
            base = os.getcwd()
            target_dir = os.path.normpath(os.path.join(base, rel_dir.lstrip("/")))
            if not target_dir.startswith(base):
                self.send_error(403, "Forbidden")
                return
            try:
                entries = os.listdir(target_dir)
            except OSError:
                entries = []
            folders = natsorted(
                e for e in entries
                if os.path.isdir(os.path.join(target_dir, e)) and not e.startswith('.')
            )
            data = json.dumps({"folders": folders}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            self.send_error(500, f"Internal error: {e}")

    # ------------------------------------------------------------------
    # /__concat__
    # ------------------------------------------------------------------

    def _serve_concat_image(self, qs):
        try:
            rel_dir = qs.get("dir", [""])[0]
            start   = int(qs.get("start", [0])[0])
            count   = int(qs.get("count", [1])[0])

            base = os.getcwd()
            target_dir = os.path.normpath(os.path.join(base, rel_dir.lstrip("/")))
            if not target_dir.startswith(base):
                self.send_error(403, "Forbidden")
                return

            all_files = _get_image_files(target_dir)
            chunk = all_files[start:start + count]

            images = []
            for fname in chunk:
                try:
                    img = Image.open(os.path.join(target_dir, fname)).convert("RGB")
                    images.append(img)
                except Exception:
                    pass

            if not images:
                self.send_error(404, "No images found")
                return

            target_h = images[0].height
            resized = []
            for img in images:
                if img.height != target_h:
                    ratio = target_h / img.height
                    img = img.resize((int(img.width * ratio), target_h), Image.LANCZOS)
                resized.append(img)

            total_w = sum(img.width for img in resized)
            combined = Image.new("RGB", (total_w, target_h))
            x = 0
            for img in resized:
                combined.paste(img, (x, 0))
                x += img.width

            buf = io.BytesIO()
            combined.save(buf, format="JPEG", quality=85)
            buf.seek(0)
            data = buf.read()

            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        except Exception as e:
            self.send_error(500, f"Internal error: {e}")


    # ------------------------------------------------------------------
    # /__multiconcat__
    # ------------------------------------------------------------------

    def _serve_multiconcat_image(self, qs):
        try:
            rel_dirs = qs.get("dir", [])
            start    = int(qs.get("start", [0])[0])
            count    = int(qs.get("count", [1])[0])

            base = os.getcwd()
            images = []
            for rel_dir in rel_dirs:
                target_dir = os.path.normpath(os.path.join(base, rel_dir.lstrip("/")))
                if not target_dir.startswith(base):
                    continue
                all_files = _get_image_files(target_dir)
                chunk = all_files[start:start + count]
                for fname in chunk:
                    try:
                        img = Image.open(os.path.join(target_dir, fname)).convert("RGB")
                        images.append(img)
                    except Exception:
                        pass

            if not images:
                self.send_error(404, "No images found")
                return

            target_h = images[0].height
            resized = []
            for img in images:
                if img.height != target_h:
                    ratio = target_h / img.height
                    img = img.resize((int(img.width * ratio), target_h), Image.LANCZOS)
                resized.append(img)

            total_w = sum(img.width for img in resized)
            combined = Image.new("RGB", (total_w, target_h))
            x = 0
            for img in resized:
                combined.paste(img, (x, 0))
                x += img.width

            buf = io.BytesIO()
            combined.save(buf, format="JPEG", quality=85)
            buf.seek(0)
            data = buf.read()

            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        except Exception as e:
            self.send_error(500, f"Internal error: {e}")


    # ------------------------------------------------------------------
    # /__vstack__
    # ------------------------------------------------------------------

    def _serve_vstack_image(self, qs):
        try:
            rel_dirs = qs.get("dir", [])
            start    = int(qs.get("start", [0])[0])
            count    = int(qs.get("count", [1])[0])

            base = os.getcwd()

            def make_strip(rel_dir):
                target_dir = os.path.normpath(os.path.join(base, rel_dir.lstrip("/")))
                if not target_dir.startswith(base):
                    return None
                all_files = _get_image_files(target_dir)
                chunk = all_files[start:start + count]
                images = []
                for fname in chunk:
                    try:
                        img = Image.open(os.path.join(target_dir, fname)).convert("RGB")
                        images.append(img)
                    except Exception:
                        pass
                if not images:
                    return None
                target_h = images[0].height
                resized = []
                for img in images:
                    if img.height != target_h:
                        ratio = target_h / img.height
                        img = img.resize((int(img.width * ratio), target_h), Image.LANCZOS)
                    resized.append(img)
                total_w = sum(img.width for img in resized)
                strip = Image.new("RGB", (total_w, target_h))
                x = 0
                for img in resized:
                    strip.paste(img, (x, 0))
                    x += img.width
                return strip

            strips = [s for s in (make_strip(d) for d in rel_dirs) if s is not None]
            if not strips:
                self.send_error(404, "No images found")
                return

            target_w = strips[0].width
            scaled = []
            for strip in strips:
                if strip.width != target_w:
                    ratio = target_w / strip.width
                    strip = strip.resize((target_w, int(strip.height * ratio)), Image.LANCZOS)
                scaled.append(strip)

            total_h = sum(s.height for s in scaled)
            combined = Image.new("RGB", (target_w, total_h))
            y = 0
            for strip in scaled:
                combined.paste(strip, (0, y))
                y += strip.height

            buf = io.BytesIO()
            combined.save(buf, format="JPEG", quality=85)
            buf.seek(0)
            data = buf.read()

            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        except Exception as e:
            self.send_error(500, f"Internal error: {e}")

    # ------------------------------------------------------------------
    # Directory listing / gallery dispatcher
    # ------------------------------------------------------------------

    def list_directory(self, path):
        try:
            file_list = natsorted(os.listdir(path))
        except OSError:
            self.send_error(404, "No permission to list directory")
            return None

        image_files = [
            f for f in file_list
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.gif'))
        ]

        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)

        concat_param = qs.get("concat", [None])[0]
        concat_n = (
            int(concat_param)
            if concat_param and concat_param.lstrip('-').isdigit() and int(concat_param) > 1
            else None
        )

        push_list = _parse_float_list(qs.get("push_list", [""])[0])
        pull_list = _parse_float_list(qs.get("pull_list", [""])[0])

        size_param = qs.get("size", [None])[0]
        img_size = int(size_param) if size_param and size_param.isdigit() and int(size_param) > 0 else None

        compare = qs.get("compare", ["0"])[0] == "1"

        display_pretrained = qs.get("display_pretrained", ["0"])[0] == "1"
        show_paths = qs.get("show_paths", ["0"])[0] == "1"
        concept_override = qs.get("concept", [None])[0] or None
        prompt_override = qs.get("prompt", [None])[0] or None
        token_sel_list = _parse_str_list(qs.get("token_sel_list", [""])[0])

        if image_files:
            return self.generate_gallery(path, image_files, concat_n, push_list, pull_list,
                                         img_size, compare, display_pretrained, show_paths,
                                         concept_override, prompt_override, token_sel_list)

        # ---- Plain directory listing (no images) ----
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()

        displaypath = unquote(self.path)
        self.wfile.write(f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Directory listing for {displaypath}</title>
<style>
    body {{ background-color: #111; color: #eee; font-family: sans-serif; }}
    a {{ color: #5fd7ff; text-decoration: none; }}
</style>
</head><body>
<h2>Directory listing for {displaypath}</h2><hr><ul>
""".encode("utf-8"))

        parent = self._parent_href()
        if parent:
            self.wfile.write(f'<li><a href="{parent}">⬅️ Go back</a></li>\n'.encode("utf-8"))

        for name in file_list:
            fullname = os.path.join(path, name)
            displayname = linkname = name
            if os.path.isdir(fullname):
                displayname = linkname = name + "/"
            self.wfile.write(f'<li><a href="{linkname}">{displayname}</a></li>\n'.encode("utf-8"))

        self.wfile.write(FUZZY_PICKER_JS.encode("utf-8"))
        self.wfile.write(b"</ul><hr></body></html>")
        return None

    # ------------------------------------------------------------------
    # Gallery renderer
    # ------------------------------------------------------------------

    def generate_gallery(self, path, image_files, concat_n=None,
                         push_list=None, pull_list=None, img_size=None,
                         compare=False, display_pretrained=False, show_paths=False,
                         concept_override=None, prompt_override=None, token_sel_list=None):
        push_list = push_list or []
        pull_list = pull_list or []
        token_sel_list = token_sel_list or []

        encoded_path = unquote(self.path)
        display_path = urlparse(encoded_path).path

        path_parts = [s for s in display_path.split('/') if s]
        exp_idx = _find_exp_segment_idx(path_parts)
        push_val, pull_val = (
            _parse_ate_ir(path_parts[exp_idx]) if exp_idx is not None else (None, None)
        )

        concept_val   = _parse_unlearned_concept(path_parts[exp_idx]) if exp_idx is not None else None
        token_val     = _parse_token_selection(path_parts[exp_idx]) if exp_idx is not None else None
        prompt_val    = path_parts[-2] if len(path_parts) >= 2 else None

        if concept_override and concept_override != concept_val:
            display_path = _rewrite_concept(display_path, concept_override)
            concept_val = concept_override
        if prompt_override and prompt_override != prompt_val:
            display_path = _rewrite_prompt(display_path, prompt_override, exp_idx)
            prompt_val = prompt_override
        path_parts = [s for s in display_path.split('/') if s]
        exp_idx = _find_exp_segment_idx(path_parts)

        has_pp = push_val is not None or pull_val is not None
        push_display = f"{push_val:.2f}" if push_val is not None else ""
        pull_display = f"{pull_val:.2f}" if pull_val is not None else ""
        push_list_display = ",".join(f"{v:.2f}" for v in push_list)
        pull_list_display = ",".join(f"{v:.2f}" for v in pull_list)

        base_cwd = os.getcwd()

        def make_rows():
            if push_list and pull_list:
                pp_combos = [(p, q) for p in push_list for q in pull_list]
            elif push_list:
                pp_combos = [(p, pull_val) for p in push_list]
            elif pull_list:
                pp_combos = [(push_val, q) for q in pull_list]
            else:
                pp_combos = []

            ts_values = token_sel_list if token_sel_list else [token_val]

            if not pp_combos and not token_sel_list:
                return []

            if pp_combos:
                combos = [(p, q, ts) for (p, q) in pp_combos for ts in ts_values]
            else:
                combos = [(push_val, pull_val, ts) for ts in ts_values]

            rows = []
            for (p, q, ts) in combos:
                new_path = _rewrite_path(display_path, p, q)
                if ts is not None:
                    new_path = _rewrite_token_selection(new_path, ts)
                rel = new_path.lstrip("/")
                abs_dir = os.path.normpath(os.path.join(base_cwd, rel))
                if not abs_dir.startswith(base_cwd):
                    continue
                imgs = _get_image_files(abs_dir)
                label_parts = []
                if push_list:
                    label_parts.append(f"push={p:.2f}")
                if pull_list:
                    label_parts.append(f"pull={q:.2f}")
                if token_sel_list:
                    label_parts.append(f"tok={ts}")
                rows.append((" | ".join(label_parts), rel, imgs))
            return rows

        multi_rows = make_rows()
        multi_mode = bool(multi_rows)

        pretrained_rel_dir = None
        pretrained_imgs = []
        if multi_mode and has_pp:
            pretrained_rel_dir = _build_pretrained_rel_dir(display_path, base_cwd)
            if pretrained_rel_dir:
                abs_pt = os.path.normpath(os.path.join(base_cwd, pretrained_rel_dir))
                if abs_pt.startswith(base_cwd):
                    pretrained_imgs = _get_image_files(abs_pt)

        can_show_pretrained = multi_mode and has_pp and pretrained_rel_dir is not None

        def build_url(extra: dict):
            params = {}
            if concat_n:
                params["concat"] = concat_n
            if push_list:
                params["push_list"] = push_list_display
            if pull_list:
                params["pull_list"] = pull_list_display
            if img_size:
                params["size"] = img_size
            if compare:
                params["compare"] = 1
            if display_pretrained:
                params["display_pretrained"] = 1
            if show_paths:
                params["show_paths"] = 1
            if token_sel_list:
                params["token_sel_list"] = ",".join(token_sel_list)
            if concept_override:
                params["concept"] = concept_override
            if prompt_override:
                params["prompt"] = prompt_override
            params.update(extra)
            params = {k: v for k, v in params.items() if v not in ("", None, 0, False)}
            return display_path + ("?" + urlencode(params) if params else "")

        current_concat = concat_n if concat_n else ""
        parent = self._parent_href()
        rel_dir = display_path.lstrip("/")

        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()

        pretrained_hint = f"→ {pretrained_rel_dir}" if pretrained_rel_dir else ""
        pretrained_found = bool(pretrained_imgs)
        pretrained_badge = (
            f'<span class="badge badge-pt">pretrained: {len(pretrained_imgs)} imgs</span>'
            if (can_show_pretrained and display_pretrained and pretrained_found)
            else (
                f'<span class="badge badge-pt-missing" title="{pretrained_hint}">pretrained: not found</span>'
                if (can_show_pretrained and display_pretrained and not pretrained_found)
                else ""
            )
        )

        self.wfile.write(f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Image Gallery</title>
<style>
    body {{ background-color: #111; color: #eee; font-family: sans-serif; padding: 12px; }}
    h2 {{ margin: 0 0 10px; font-size: 14px; color: #888; word-break: break-all; }}
    .toolbar {{ margin-bottom: 14px; display: flex; align-items: flex-start; gap: 14px; flex-wrap: wrap; }}
    .tg {{ display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }}
    label {{ color: #ccc; font-size: 13px; }}
    .hint {{ font-size: 11px; color: #666; }}
    input[type=number], input[type=text] {{
        padding: 4px 6px; border-radius: 4px;
        border: 1px solid #555; background: #222; color: #eee; font-size: 13px;
    }}
    input[type=number] {{ width: 60px; }}
    input[type=text]   {{ width: 150px; }}
    button {{
        padding: 4px 12px; border-radius: 4px; border: none;
        background: #5fd7ff; color: #111; cursor: pointer; font-weight: bold; font-size: 13px;
    }}
    button:hover {{ background: #38b6d8; }}
    button.dim {{ background: #444; color: #bbb; }}
    button.pt-on {{ background: #f5a623; color: #111; }}
    button.pt-on:hover {{ background: #d48a10; }}
    .sep {{ color: #444; font-size: 20px; align-self: center; }}
    .badge {{
        background: #2a2a2a; border: 1px solid #444; border-radius: 4px;
        padding: 3px 8px; font-size: 12px; color: #888;
    }}
    .badge-pt {{
        background: #1e1500; border: 1px solid #f5a623; border-radius: 4px;
        padding: 3px 8px; font-size: 12px; color: #f5a623;
    }}
    .badge-pt-missing {{
        background: #1e1500; border: 1px solid #666; border-radius: 4px;
        padding: 3px 8px; font-size: 12px; color: #666;
    }}
    .grid {{ display: grid; grid-template-columns: repeat(10, 1fr); gap: 6px; margin-bottom: 6px; }}
    .grid.concat-mode {{ grid-template-columns: 1fr; }}
    .item {{ text-align: center; font-size: 11px; word-break: break-all; }}
    img {{ width: 100%; height: auto; border-radius: 3px; display: block; }}
    .sized img {{ width: auto; height: var(--img-h); max-width: 100%; }}
    .row-block {{ margin-bottom: 20px; }}
    .row-label {{
        font-size: 13px; font-weight: bold; color: #5fd7ff;
        margin-bottom: 6px; padding: 4px 10px;
        background: #0e1e28; border-left: 3px solid #5fd7ff; border-radius: 2px;
        display: inline-block;
    }}
    .row-label-pt {{
        font-size: 13px; font-weight: bold; color: #f5a623;
        margin-bottom: 6px; padding: 4px 10px;
        background: #1e1500; border-left: 3px solid #f5a623; border-radius: 2px;
        display: inline-block;
    }}
    .row-empty {{ color: #555; font-size: 12px; padding: 4px 10px; font-style: italic; }}
    .compare-block {{ margin-bottom: 28px; border: 1px solid #2a2a2a; border-radius: 4px; overflow: hidden; }}
    .compare-pos {{
        font-size: 12px; font-weight: bold; color: #888;
        padding: 4px 10px; background: #1a1a1a; border-bottom: 1px solid #2a2a2a;
    }}
    .compare-row {{ display: flex; flex-direction: column; border-bottom: 1px solid #1e1e1e; }}
    .compare-row:last-child {{ border-bottom: none; }}
    .compare-label {{
        font-size: 12px; font-weight: bold; color: #5fd7ff;
        padding: 3px 10px; background: #0a1820;
    }}
    .compare-row img {{ width: 100%; height: auto; display: block; }}
    .sized .compare-row img {{ width: auto; height: var(--img-h); max-width: 100%; }}
</style>
</head><body>
<h2>Gallery: {display_path}</h2>
<div class="{'sized' if img_size else ''}" style="{'--img-h:' + str(img_size) + 'px' if img_size else ''}">
<div class="toolbar">
  {'<a href="' + parent + '">⬅️ Go back</a>' if parent else ''}

  <form method="get" action="{display_path}" style="display:contents">
    {'<input type="hidden" name="push_list" value="' + push_list_display + '">' if push_list else ''}
    {'<input type="hidden" name="pull_list" value="' + pull_list_display + '">' if pull_list else ''}
    <div class="tg">
      <label>Concat every</label>
      <input type="number" name="concat" min="0" max="100" value="{current_concat}" placeholder="N">
      <button type="submit">Apply</button>
      {'<a href="' + build_url({"concat": ""}) + '"><button type="button" class="dim">✕</button></a>' if concat_n else ''}
    </div>
  </form>

  <span class="badge">{len(image_files)} images</span>
  {f'<span class="badge">concat={concat_n}</span>' if concat_n else ''}
  {f'<span class="badge">tok={token_val}</span>' if token_val else ''}
  {pretrained_badge}

  <form method="get" action="{display_path}" style="display:contents">
    {'<input type="hidden" name="concat" value="' + str(concat_n) + '">' if concat_n else ''}
    {'<input type="hidden" name="push_list" value="' + push_list_display + '">' if push_list else ''}
    {'<input type="hidden" name="pull_list" value="' + pull_list_display + '">' if pull_list else ''}
    <div class="tg">
      <label>Size</label>
      <input type="number" name="size" min="50" max="2000" step="50" value="{img_size if img_size else ''}" placeholder="px">
      <button type="submit">Apply</button>
      {'<a href="' + build_url({"size": ""}) + '"><button type="button" class="dim">✕</button></a>' if img_size else ''}
    </div>
  </form>

  {'<span class="sep">|</span>' if has_pp else ''}

  {'<form method="get" action="' + display_path + '" style="display:contents">' if has_pp else ''}
  {'<input type="hidden" name="concat" value="' + str(concat_n) + '">' if (has_pp and concat_n) else ''}
  {'<div class="tg">' if has_pp else ''}
  {'<label>push <span class="hint">(AtE)</span></label>' if has_pp else ''}
  {'<input type="text" name="push_list" value="' + push_list_display + '" placeholder="e.g. 0.20,0.40,0.60">' if has_pp else ''}
  {'<label>pull <span class="hint">(Ir)</span></label>' if has_pp else ''}
  {'<input type="text" name="pull_list" value="' + pull_list_display + '" placeholder="e.g. 0.40,0.60">' if has_pp else ''}
  {'<label>token <span class="hint">(sel)</span></label>' if has_pp else ''}
  {'<input type="text" name="token_sel_list" value="' + ",".join(token_sel_list) + '" placeholder="e.g. mce-1,mt,mE" style="width:130px">' if has_pp else ''}
  {'<button type="submit">Go</button>' if has_pp else ''}
  {'<a href="' + build_url({"push_list": "", "pull_list": "", "token_sel_list": ""}) + '"><button type="button" class="dim">✕ lists</button></a>' if (has_pp and multi_mode) else ''}
  {'</div></form>' if has_pp else ''}

  {'<span class="sep">|</span>' if exp_idx is not None else ''}
  {'<form method="get" action="' + display_path + '" style="display:contents">' if exp_idx is not None else ''}
  {'<input type="hidden" name="concat" value="' + str(concat_n) + '">' if (exp_idx is not None and concat_n) else ''}
  {'<input type="hidden" name="push_list" value="' + push_list_display + '">' if (exp_idx is not None and push_list) else ''}
  {'<input type="hidden" name="pull_list" value="' + pull_list_display + '">' if (exp_idx is not None and pull_list) else ''}
  {'<input type="hidden" name="size" value="' + str(img_size) + '">' if (exp_idx is not None and img_size) else ''}
  {'<input type="hidden" name="compare" value="1">' if (exp_idx is not None and compare) else ''}
  {'<input type="hidden" name="display_pretrained" value="1">' if (exp_idx is not None and display_pretrained) else ''}
  {'<div class="tg">' if exp_idx is not None else ''}
  {'<label>concept <span class="hint">(U.)</span></label>' if exp_idx is not None else ''}
  {'<input type="text" name="concept" value="' + (concept_val or '') + '" placeholder="e.g. shibuya" style="width:100px">' if exp_idx is not None else ''}
  {'<label>prompt</label>' if exp_idx is not None else ''}
  {'<input type="text" name="prompt" value="' + (prompt_val or '') + '" placeholder="e.g. a photo of Shibuya" style="width:200px">' if exp_idx is not None else ''}
  {'<button type="submit">Go</button>' if exp_idx is not None else ''}
  {'<a href="' + build_url({"concept": "", "prompt": ""}) + '"><button type="button" class="dim">✕</button></a>' if (exp_idx is not None and (concept_override or prompt_override)) else ''}
  {'</div></form>' if exp_idx is not None else ''}

  {('<a href="' + build_url({"compare": ""}) + '"><button type="button" class="dim">☰ separate</button></a>' if compare else '<a href="' + build_url({"compare": 1}) + '"><button type="button">⊞ compare</button></a>') if multi_mode else ''}

  {('<a href="' + build_url({"display_pretrained": ""}) + '"><button type="button" class="pt-on">🖼 hide pretrained</button></a>' if display_pretrained else '<a href="' + build_url({"display_pretrained": 1}) + '"><button type="button">🖼 show pretrained</button></a>') if can_show_pretrained else ''}

  {'<a href="' + build_url({"show_paths": ""}) + '"><button type="button" class="pt-on">🗂 hide paths</button></a>' if show_paths else '<a href="' + build_url({"show_paths": 1}) + '"><button type="button" class="dim">🗂 show paths</button></a>'}
</div>
""".encode("utf-8"))

        if show_paths and multi_mode:
            rows_for_debug = []
            if can_show_pretrained:
                abs_pt = os.path.normpath(os.path.join(base_cwd, pretrained_rel_dir)) if pretrained_rel_dir else "N/A"
                rows_for_debug.append(("pretrained", pretrained_rel_dir or "N/A", abs_pt, len(pretrained_imgs)))
            for (label, rel_dir_r, img_files_r) in multi_rows:
                abs_r = os.path.normpath(os.path.join(base_cwd, rel_dir_r))
                rows_for_debug.append((label, rel_dir_r, abs_r, len(img_files_r)))

            self.wfile.write(b'''<div style="margin-bottom:16px;background:#0a0a0a;border:1px solid #333;border-radius:4px;overflow:auto">
<table style="border-collapse:collapse;width:100%;font-size:11px;font-family:monospace">
<thead><tr style="background:#1a1a1a;color:#888">
  <th style="padding:5px 10px;text-align:left;border-bottom:1px solid #333">Label</th>
  <th style="padding:5px 10px;text-align:left;border-bottom:1px solid #333">Rel path (used in URL)</th>
  <th style="padding:5px 10px;text-align:left;border-bottom:1px solid #333">Abs path (on disk)</th>
  <th style="padding:5px 10px;text-align:right;border-bottom:1px solid #333">imgs</th>
</tr></thead><tbody>''')
            for (label, rel_p, abs_p, n_imgs) in rows_for_debug:
                color = "#f5a623" if label == "pretrained" else "#5fd7ff"
                found_color = "#4caf50" if n_imgs > 0 else "#f44336"
                self.wfile.write((
                    f'<tr style="border-bottom:1px solid #1e1e1e">' +
                    f'<td style="padding:4px 10px;color:{color};white-space:nowrap">{label}</td>' +
                    f'<td style="padding:4px 10px;color:#aaa;word-break:break-all">{rel_p}</td>' +
                    f'<td style="padding:4px 10px;color:#777;word-break:break-all">{abs_p}</td>' +
                    f'<td style="padding:4px 10px;color:{found_color};text-align:right;white-space:nowrap">{n_imgs}</td>' +
                    f'</tr>\n'
                ).encode("utf-8"))
            self.wfile.write(b'</tbody></table></div>\n')

        if multi_mode:
            self._write_multi_rows(
                multi_rows, concat_n, img_size, compare,
                display_pretrained=display_pretrained,
                pretrained_rel_dir=pretrained_rel_dir,
                pretrained_imgs=pretrained_imgs,
            )
        else:
            self._write_single_grid(image_files, concat_n, rel_dir, img_size)

        self.wfile.write(FUZZY_PICKER_JS.encode("utf-8"))
        self.wfile.write(b"</div></body></html>")
        return None

    # ------------------------------------------------------------------
    # Grid renderers
    # ------------------------------------------------------------------

    def _write_single_grid(self, image_files, concat_n, rel_dir, img_size=None):
        if concat_n:
            self.wfile.write(b'<div class="grid concat-mode">\n')
            i = 0
            while i < len(image_files):
                count = min(concat_n, len(image_files) - i)
                src = f"/__concat__?dir={rel_dir}&start={i}&count={count}"
                label = f"Images {i+1}–{i+count}"
                self.wfile.write(f'<div class="item"><img src="{src}" alt="{label}" loading="lazy"><div>{label}</div></div>\n'.encode("utf-8"))
                i += concat_n
        else:
            self.wfile.write(b'<div class="grid">\n')
            for fname in image_files:
                self.wfile.write(f'<div class="item"><img src="{fname}" alt="{fname}" loading="lazy"><div>{fname}</div></div>\n'.encode("utf-8"))
        self.wfile.write(b"</div>\n")

    def _write_multi_rows(self, rows, concat_n, img_size=None, compare=False,
                          display_pretrained=False, pretrained_rel_dir=None, pretrained_imgs=None):
        pretrained_imgs = pretrained_imgs or []
        swap_n = concat_n if concat_n else 1
        use_concat = bool(concat_n)

        max_imgs = max((len(imgs) for (_, _, imgs) in rows), default=0)
        if display_pretrained and pretrained_imgs:
            max_imgs = max(max_imgs, len(pretrained_imgs))

        if max_imgs == 0:
            self.wfile.write(b'<div class="row-empty">&#9888; no images found in any variant</div>\n')
            return

        pos = 0
        while pos < max_imgs:
            if compare:
                count = swap_n
                img_range = f"#{pos+1}" if swap_n == 1 else f"#{pos+1}&#8211;{pos+count}"

                dir_params_list = []
                if display_pretrained and pretrained_rel_dir and pos < len(pretrained_imgs):
                    dir_params_list.append(f"dir={pretrained_rel_dir}")
                for (_, rel_dir, img_files) in rows:
                    if pos < len(img_files):
                        dir_params_list.append(f"dir={rel_dir}")

                if not dir_params_list:
                    pos += swap_n
                    continue

                dir_params = "&".join(dir_params_list)
                src = f"/__vstack__?{dir_params}&start={pos}&count={count}"

                self.wfile.write(b'<div class="row-block">\n')
                self.wfile.write(f'<div class="row-label">{img_range}</div>\n'.encode("utf-8"))
                self.wfile.write((
                    f'<div class="grid concat-mode">'
                    f'<div class="item"><img src="{src}" loading="lazy"></div>'
                    f'</div>\n'
                ).encode("utf-8"))
                self.wfile.write(b"</div>\n")

            else:
                if display_pretrained and pretrained_rel_dir:
                    if pos < len(pretrained_imgs):
                        count = min(swap_n, len(pretrained_imgs) - pos)
                        img_range = f"#{pos+1}" if swap_n == 1 else f"#{pos+1}&#8211;{pos+count}"
                        full_label = f"pretrained &nbsp;&middot;&nbsp; {img_range}"

                        self.wfile.write(b'<div class="row-block">\n')
                        self.wfile.write(f'<div class="row-label-pt">{full_label}</div>\n'.encode("utf-8"))

                        if use_concat:
                            src = f"/__concat__?dir={pretrained_rel_dir}&start={pos}&count={count}"
                            self.wfile.write((
                                f'<div class="grid concat-mode">'
                                f'<div class="item"><img src="{src}" loading="lazy"></div>'
                                f'</div>\n'
                            ).encode("utf-8"))
                        else:
                            fname = pretrained_imgs[pos]
                            src = "/" + pretrained_rel_dir.rstrip("/") + "/" + fname
                            self.wfile.write((
                                f'<div class="grid">'
                                f'<div class="item" style="grid-column:span 1">'
                                f'<img src="{src}" loading="lazy">'
                                f'<div>{fname}</div>'
                                f'</div>'
                                f'</div>\n'
                            ).encode("utf-8"))

                        self.wfile.write(b"</div>\n")
                    else:
                        img_range = f"#{pos+1}" if swap_n == 1 else f"#{pos+1}&#8211;{pos+swap_n}"
                        self.wfile.write((
                            f'<div class="row-block">'
                            f'<div class="row-label-pt">pretrained &nbsp;&middot;&nbsp; {img_range}</div>'
                            f'<div class="row-empty">&#9888; no image at this position</div>'
                            f'</div>\n'
                        ).encode("utf-8"))

                for (label, rel_dir, img_files) in rows:
                    if pos >= len(img_files):
                        img_range = f"#{pos+1}" if swap_n == 1 else f"#{pos+1}&#8211;{pos+swap_n}"
                        self.wfile.write((
                            f'<div class="row-block">'
                            f'<div class="row-label">{label} &nbsp;&middot;&nbsp; {img_range}</div>'
                            f'<div class="row-empty">&#9888; no image at this position</div>'
                            f'</div>\n'
                        ).encode("utf-8"))
                        continue

                    count = min(swap_n, len(img_files) - pos)
                    img_range = f"#{pos+1}" if swap_n == 1 else f"#{pos+1}&#8211;{pos+count}"
                    full_label = f"{label} &nbsp;&middot;&nbsp; {img_range}"

                    self.wfile.write(b'<div class="row-block">\n')
                    self.wfile.write(f'<div class="row-label">{full_label}</div>\n'.encode("utf-8"))

                    if use_concat:
                        src = f"/__concat__?dir={rel_dir}&start={pos}&count={count}"
                        self.wfile.write((
                            f'<div class="grid concat-mode">'
                            f'<div class="item"><img src="{src}" loading="lazy"></div>'
                            f'</div>\n'
                        ).encode("utf-8"))
                    else:
                        fname = img_files[pos]
                        src = "/" + rel_dir.rstrip("/") + "/" + fname
                        self.wfile.write((
                            f'<div class="grid">'
                            f'<div class="item" style="grid-column:span 1">'
                            f'<img src="{src}" loading="lazy">'
                            f'<div>{fname}</div>'
                            f'</div>'
                            f'</div>\n'
                        ).encode("utf-8"))

                    self.wfile.write(b"</div>\n")

            pos += swap_n


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_server(port=8080, root_dir="."):
    os.chdir(root_dir)
    server = ThreadingHTTPServer(("", port), GalleryHTTPRequestHandler)
    print(f"Serving on http://localhost:{port}/")
    server.serve_forever()


if __name__ == "__main__":
    run_server(port=8080, root_dir=".")