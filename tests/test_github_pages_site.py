"""Tests for the GitHub Pages Complex Dynamics showcase site and deployment workflow."""

from __future__ import annotations

from html.parser import HTMLParser
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
ROOT_INDEX = REPO / "index.html"
SUB_INDEX = REPO / "experiments" / "strogatz-dynamics" / "index.html"
NOJEKYLL = REPO / ".nojekyll"
PAGES_WORKFLOW = REPO / ".github" / "workflows" / "pages.yml"
MEDIA_DIR = REPO / "experiments" / "strogatz-dynamics" / "media"


def _node_binary() -> str | None:
    return shutil.which("node") or (
        "/opt/homebrew/bin/node" if Path("/opt/homebrew/bin/node").exists() else None
    )


def test_pages_entrypoints_and_zero_external_runtime_dependencies() -> None:
    assert ROOT_INDEX.is_file()
    assert SUB_INDEX.is_file()
    assert NOJEKYLL.is_file()
    assert PAGES_WORKFLOW.is_file()

    html = ROOT_INDEX.read_text(encoding="utf-8")
    sub_html = SUB_INDEX.read_text(encoding="utf-8")

    # Zero external CDN scripts, stylesheets, fonts, images, videos, or iframes
    forbidden_external_attr = re.compile(
        r'<(?:script|link|img|video|source|iframe)\b[^>]*(?:src|href|poster)=["\']https?://',
        re.IGNORECASE,
    )
    assert not forbidden_external_attr.search(html), "External runtime dependency found in index.html"
    assert not forbidden_external_attr.search(sub_html), "External runtime dependency found in sub-index"
    assert "@import" not in html

    # Static HTML attributes outside <script> blocks must resolve on disk
    for page_path, content in ((ROOT_INDEX, html), (SUB_INDEX, sub_html)):
        markup_only = re.sub(r"<script\b[^>]*>[\s\S]*?</script>", "", content, flags=re.IGNORECASE)
        attrs = re.findall(r'(?:href|src|poster)=["\']([^"\']+)["\']', markup_only)
        for target in attrs:
            if target.startswith(("#", "https://github.com/dstadulis/Braitenberg-Vehicles")):
                continue
            clean = target.split("?", 1)[0].split("#", 1)[0]
            if not clean:
                continue
            resolved = (page_path.parent / clean).resolve()
            assert resolved.is_file(), f"Broken HTML link in {page_path.name}: {target} -> {resolved}"

    # Verify balanced HTML tags, unique IDs, valid ARIA targets, and exact JS getElementById bindings
    void_tags = {
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr",
    }

    class _DomAuditor(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.stack: list[tuple[str, tuple[int, int]]] = []
            self.ids: list[str] = []
            self.labelled_by: list[str] = []

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            ad = {k: (v or "") for k, v in attrs}
            if ad.get("id"):
                self.ids.append(ad["id"])
            if ad.get("aria-labelledby"):
                self.labelled_by.extend(ad["aria-labelledby"].split())
            if tag not in void_tags:
                self.stack.append((tag, self.getpos()))

        def handle_endtag(self, tag: str) -> None:
            if tag in void_tags:
                return
            assert self.stack, f"Unexpected closing </{tag}> at {self.getpos()}"
            open_tag, pos = self.stack.pop()
            assert open_tag == tag, f"Mismatched </{tag}> at {self.getpos()} for <{open_tag}> at {pos}"

    markup_only = re.sub(r"<script\b[^>]*>[\s\S]*?</script>", "", html, flags=re.IGNORECASE)
    auditor = _DomAuditor()
    auditor.feed(markup_only)
    assert not auditor.stack, f"Unclosed HTML tags in index.html: {auditor.stack}"

    dup_ids = [i for i in set(auditor.ids) if auditor.ids.count(i) > 1]
    assert not dup_ids, f"Duplicate HTML IDs in index.html: {dup_ids}"
    assert not (set(auditor.labelled_by) - set(auditor.ids))

    js_code = re.search(r"<script>([\s\S]*?)</script>", html).group(1)
    doc_ids = set(re.findall(r"\bdocument\.getElementById\([\"']([^\"']+)[\"']\)", js_code))
    assert not (doc_ids - set(auditor.ids)), f"Missing DOM IDs in index.html: {doc_ids - set(auditor.ids)}"

    explorer_html = (REPO / "experiments" / "strogatz-dynamics" / "explorer.html").read_text(encoding="utf-8")
    explorer_ids = set(re.findall(r'\bid=["\']([^"\']+)["\']', explorer_html))
    iframe_ids = set(re.findall(r"contentDocument\.getElementById\([\"']([^\"']+)[\"']\)", js_code))
    assert not (iframe_ids - explorer_ids), f"Missing explorer.html DOM IDs: {iframe_ids - explorer_ids}"

    # Sub-index meta refresh must live inside <noscript> so JS query/hash forwarding cannot race it
    assert "<noscript><meta http-equiv=\"refresh\"" in sub_html

    # Responsive CSS guards: no 400px mobile dead-space min-height on .cinema-viewport and
    # aspect-ratio rules resolve height:auto from width:min(...) to prevent 1:1 / 16:9 stretching
    style_block = re.search(r"<style>([\s\S]*?)</style>", html).group(1)
    assert "min-height: 400px" not in style_block
    assert 'width: min(100%, 660px, calc(68vh - 16px));' in style_block
    assert "minmax(0, 1.45fr)" in style_block
    assert "minmax(0, 1.15fr)" in style_block


def test_showcase_js_catalog_covers_all_78_media_assets_and_runtime_controls(tmp_path: Path) -> None:
    node_bin = _node_binary()
    if node_bin is None:
        pytest.skip("node binary not installed")

    html = ROOT_INDEX.read_text(encoding="utf-8")
    script_match = re.search(r"<script>([\s\S]*?)</script>", html)
    assert script_match is not None
    js_code = script_match.group(1)

    # 1. JS syntax check
    syntax_file = tmp_path / "index_syntax_check.js"
    syntax_file.write_text(js_code, encoding="utf-8")
    syntax = subprocess.run([node_bin, "--check", str(syntax_file)], capture_output=True, text=True)
    assert syntax.returncode == 0, syntax.stderr

    # 2. Full runtime harness exercising initial state, URL/blob resolution, 30 MP4 sweeps,
    #    38 4K plate theme swaps, gallery filters, lightbox navigation, and error/autoplay edge cases
    harness_file = tmp_path / "index_runtime_verify.js"
    harness_file.write_text(
        """
function makeEl(tag, id = "") {
  return {
    tagName: (tag || "DIV").toUpperCase(),
    id,
    className: "",
    textContent: "",
    value: "1",
    href: "",
    src: "",
    poster: "",
    alt: "",
    muted: true,
    paused: false,
    playsInline: true,
    playbackRate: 1.0,
    currentTime: 0.0,
    duration: 7.0,
    style: { display: "block" },
    attrs: {},
    children: [],
    listeners: {},
    _innerHTML: "",
    set innerHTML(val) {
      this._innerHTML = val;
      this.children = [];
      // Parse data-plate-idx / data-scene-thumb / data-open-lb / data-plate-video / data-plate-lab / data-matrix-play children
      const imgRe = /<img\\s+([^>]+)>/g;
      let m;
      while ((m = imgRe.exec(val)) !== null) {
        const child = makeEl("IMG");
        const attrRe = /([a-zA-Z0-9_-]+)=["']([^"']*)["']/g;
        let am;
        while ((am = attrRe.exec(m[1])) !== null) {
          child.attrs[am[1]] = am[2];
          if (am[1] === "src") child.src = am[2];
        }
        this.children.push(child);
      }
      const btnRe = /<(button|div)\\s+([^>]+)>/g;
      while ((m = btnRe.exec(val)) !== null) {
        const child = makeEl(m[1]);
        const attrRe = /([a-zA-Z0-9_-]+)=["']([^"']*)["']/g;
        let am;
        while ((am = attrRe.exec(m[2])) !== null) {
          child.attrs[am[1]] = am[2];
        }
        this.children.push(child);
      }
    },
    get innerHTML() { return this._innerHTML; },
    setAttribute(k, v) {
      this.attrs[k] = String(v);
      if (k === "src") this.src = String(v);
      if (k === "href") this.href = String(v);
    },
    getAttribute(k) { return this.attrs[k] ?? (k === "src" ? this.src : null); },
    appendChild(ch) { this.children.push(ch); return ch; },
    removeChild() {},
    select() {},
    scrollIntoView() {},
    load() {},
    play() { this.paused = false; return Promise.resolve(); },
    pause() { this.paused = true; },
    click() { if (this.listeners.click) this.listeners.click({ preventDefault() {}, target: this }); },
    addEventListener(ev, fn) { this.listeners[ev] = fn; },
    querySelectorAll(sel) {
      const out = [];
      const visit = node => {
        for (const c of node.children) {
          if (sel === "[data-open-lb]" && c.attrs["data-open-lb"] !== undefined) out.push(c);
          if (sel === "button[data-plate-video]" && c.tagName === "BUTTON" && c.attrs["data-plate-video"] !== undefined) out.push(c);
          if (sel === "button[data-plate-lab]" && c.tagName === "BUTTON" && c.attrs["data-plate-lab"] !== undefined) out.push(c);
          if (sel === "button[data-matrix-play]" && c.tagName === "BUTTON" && c.attrs["data-matrix-play"] !== undefined) out.push(c);
          visit(c);
        }
      };
      visit(this);
      return out;
    },
    classList: {
      _set: new Set(),
      add(c) { this._set.add(c); },
      remove(c) { this._set.delete(c); },
      toggle(c, force) {
        if (force === undefined) {
          if (this._set.has(c)) { this._set.delete(c); return false; }
          this._set.add(c); return true;
        }
        if (force) this._set.add(c); else this._set.delete(c);
        return Boolean(force);
      },
      contains(c) { return this._set.has(c); }
    }
  };
}

const byId = {};
function getEl(id) {
  if (!byId[id]) byId[id] = makeEl("DIV", id);
  return byId[id];
}

// Pre-create segmented aspect, gallery-filter, and lab-preset buttons
const aspectButtons = ["16x9", "1x1", "9x16"].map(a => {
  const b = makeEl("BUTTON"); b.setAttribute("data-aspect-btn", a); return b;
});
const filterButtons = ["all", "phase_portraits", "tripanels", "phase_atlas"].map(f => {
  const b = makeEl("BUTTON"); b.setAttribute("data-gallery-filter", f); return b;
});
const labButtons = [0, 1, 2, 3, 4, 5].map(i => {
  const b = makeEl("BUTTON"); b.setAttribute("data-lab-preset", String(i)); return b;
});

const rootDocEl = makeEl("HTML");
rootDocEl.setAttribute("data-theme", "cosmic");

const document = {
  documentElement: rootDocEl,
  body: makeEl("BODY"),
  execCommand() { return true; },
  createElement(tag) { return makeEl(tag); },
  getElementById(id) { return getEl(id); },
  querySelectorAll(sel) {
    if (sel === "button[data-aspect-btn]") return aspectButtons;
    if (sel === "button[data-gallery-filter]") return filterButtons;
    if (sel === "button[data-lab-preset]") return labButtons;
    if (sel === ".scene-btn") {
      return [...getEl("playlistAtlas").children, ...getEl("playlistTripanels").children];
    }
    if (sel === "img[data-scene-thumb]") {
      const imgs = [];
      for (const btn of [...getEl("playlistAtlas").children, ...getEl("playlistTripanels").children]) {
        for (const ch of btn.children) {
          if (ch.attrs["data-scene-thumb"] !== undefined) imgs.push(ch);
        }
      }
      return imgs;
    }
    if (sel === "img[data-plate-idx]") {
      const imgs = [];
      for (const card of getEl("platesGrid").children) {
        for (const ch of card.children) {
          if (ch.attrs["data-plate-idx"] !== undefined) imgs.push(ch);
        }
      }
      return imgs;
    }
    return [];
  }
};

const localStorage = {
  _s: {},
  getItem(k) { return this._s[k] ?? null; },
  setItem(k, v) { this._s[k] = String(v); }
};

const window = {
  location: {
    protocol: "https:",
    pathname: "/Braitenberg-Vehicles/",
    search: "",
    hash: "",
    href: "https://dstadulis.github.io/Braitenberg-Vehicles/"
  },
  history: {
    lastUrl: "",
    replaceState(_s, _t, url) { this.lastUrl = url; }
  },
  listeners: {},
  addEventListener(ev, fn) { this.listeners[ev] = fn; }
};
const navigator = { clipboard: { writeText: () => Promise.resolve() } };
"""
        + js_code
        + """
(async () => {
  const out = {};

  // 1. Default initial video must be daydreamer_16x9.mp4
  out.initialVideoSrc = getEl("cinemaVideo").src;
  out.initialPoster = getEl("cinemaVideo").poster;
  out.initialTheme = document.documentElement.getAttribute("data-theme");

  // 2. Collect all 30 MP4s across 10 scenes x 3 aspects, plus GIFs and Posters
  const visitedMp4s = [];
  const sceneGifs = [];
  const scenePosters = [];
  for (let i = 0; i < VIDEO_SCENES.length; i++) {
    sceneGifs.push(VIDEO_SCENES[i].gif);
    scenePosters.push(VIDEO_SCENES[i].posterCosmic, VIDEO_SCENES[i].posterTextbook);
    for (const asp of ["16x9", "1x1", "9x16"]) {
      selectScene(i, asp, true);
      visitedMp4s.push(getEl("cinemaVideo").src);
    }
  }
  out.visitedMp4s = visitedMp4s;
  out.sceneGifs = sceneGifs;

  // 3. Theme toggle flips all 19 plate images between cosmic and textbook (38 PNGs)
  applyTheme("cosmic");
  const cosmicPlateSrcs = document.querySelectorAll("img[data-plate-idx]").map(im => im.src);
  applyTheme("textbook");
  const textbookPlateSrcs = document.querySelectorAll("img[data-plate-idx]").map(im => im.src);
  out.cosmicPlateSrcs = cosmicPlateSrcs;
  out.textbookPlateSrcs = textbookPlateSrcs;
  out.themeAfterFlip = document.documentElement.getAttribute("data-theme");

  // 4. Gallery filters count check
  const filterCounts = {};
  for (const btn of filterButtons) {
    btn.click();
    filterCounts[btn.getAttribute("data-gallery-filter")] = getEl("platesGrid").children.length;
  }
  filterButtons[0].click();
  out.filterCounts = filterCounts;

  // 5. Lightbox open / next / prev / close
  openLightbox(0);
  const lb0 = getEl("lbImage").src;
  stepLightbox(1);
  const lb1 = getEl("lbImage").src;
  stepLightbox(-1);
  const lbBack = getEl("lbImage").src;
  closeLightbox();
  out.lightboxOk = (lb0 !== lb1 && lb0 === lbBack && appState.lightboxIndex === -1);

  // 6. GitHub /blob/, /raw/, branch-with-slash, share-link, and edge-case resolver tests
  out.resolveBlobDaydreamer = resolveVideoInput(
    "https://github.com/dstadulis/Braitenberg-Vehicles/blob/main/experiments/strogatz-dynamics/media/mp4/phase_atlas/daydreamer_16x9.mp4"
  );
  out.resolveBlobWithRawQuery = resolveVideoInput(
    "https://github.com/dstadulis/Braitenberg-Vehicles/blob/main/experiments/strogatz-dynamics/media/mp4/phase_atlas/daydreamer_16x9.mp4?raw=true"
  );
  out.resolveRawRefsHeads = resolveVideoInput(
    "https://github.com/dstadulis/Braitenberg-Vehicles/raw/refs/heads/main/experiments/strogatz-dynamics/media/mp4/phase_atlas/daydreamer_16x9.mp4"
  );
  out.resolveRawUserContent = resolveVideoInput(
    "https://raw.githubusercontent.com/dstadulis/Braitenberg-Vehicles/main/experiments/strogatz-dynamics/media/mp4/v9_shy_9x16.mp4"
  );
  out.resolveSlashBranch = resolveVideoInput(
    "https://github.com/dstadulis/Braitenberg-Vehicles/blob/feat/complex-dynamics/experiments/strogatz-dynamics/media/mp4/v5_moth_1x1.mp4"
  );
  out.resolveMonographGif = resolveVideoInput("media/v9_shy.gif");
  out.resolveShareUrl = resolveVideoInput(
    "https://dstadulis.github.io/Braitenberg-Vehicles/?scene=v7_wary_moth&aspect=9x16"
  );
  out.resolveBlobShyVertical = resolveVideoInput(
    "https://github.com/dstadulis/Braitenberg-Vehicles/blob/main/experiments/strogatz-dynamics/media/mp4/v9_shy_9x16.mp4"
  );
  out.resolveShort = resolveVideoInput("ftle_swarm_1x1.mp4");
  out.resolveEmpty = resolveVideoInput("   ");
  out.resolveBareExtension = resolveVideoInput(".mp4");
  out.resolveBareAspect = resolveVideoInput("16x9");
  out.resolveBareRepoUrl = resolveVideoInput("https://github.com/dstadulis/Braitenberg-Vehicles/");
  out.resolveMalformedUri = resolveVideoInput("%E0%A4%A_daydreamer_16x9.mp4");
  out.resolveUnknown = resolveVideoInput("totally_nonexistent_vehicle_999");

  // 7. Query-param parsing, dynamic CLI flags & file:// share URL tests
  window.location.search = "?scene=v7_wary_moth&aspect=9x16&theme=textbook";
  parseInitialUrlParams();
  applyTheme(appState.theme, false);
  selectScene(appState.sceneIdx, appState.aspect, false);
  out.queryLoadedMp4 = getEl("cinemaVideo").src;
  out.queryLoadedTheme = document.documentElement.getAttribute("data-theme");
  out.queryCliCmd = getEl("sceneCliCmd").textContent;
  window.location.protocol = "file:";
  out.fileShareUrl = buildShareableUrl();
  window.location.protocol = "https:";

  // 8. Autoplay NotAllowedError rejection resilience + codec error fallback + GIF exit on video CTAs
  let unhandledRejected = false;
  process.on("unhandledRejection", () => { unhandledRejected = true; });
  getEl("cinemaVideo").play = () => Promise.reject(new Error("NotAllowedError"));
  selectScene(0, "16x9", true);
  await new Promise(r => setTimeout(r, 20));
  out.unhandledRejected = unhandledRejected;

  getEl("cinemaVideo").listeners.error();
  out.gifFallbackActivated = (
    appState.gifMode === true &&
    getEl("cinemaGifStage").style.display === "block" &&
    getEl("videoFallbackNotice").classList.contains("visible")
  );
  // Selecting another scene after transient error automatically retries MP4 mode
  selectScene(1, "16x9", true);
  out.gifAutoFallbackClearedOnSceneSelect = (appState.gifMode === false);

  // Manual GIF preview exits when clicking explicit 60fps video CTAs
  setGifMode(true, false);
  getEl("btnHeroPlayDaydreamer").click();
  const exitedOnHeroCta = (appState.gifMode === false);
  setGifMode(true, false);
  getEl("platesGrid").querySelectorAll("button[data-plate-video]")[0].click();
  const exitedOnPlateVideoBtn = (appState.gifMode === false);
  out.gifExitedOnExplicitVideoCtas = (exitedOnHeroCta && exitedOnPlateVideoBtn);
  setGifMode(false);
  out.gifFallbackRestored = (appState.gifMode === false && getEl("cinemaVideo").style.display === "block");

  // 9. Lazy/cross-origin iframe postMessage bridge test
  const postedLabMessages = [];
  getEl("labFrame").contentWindow = {
    postMessage(payload) { postedLabMessages.push(payload); }
  };
  launchLabPreset(4, true); // Before iframe ready (simulating lazy load or file://)
  const preLoadDirty = appState.labPresetDirty;
  const fullscreenHref = getEl("linkFullscreenLab").href;
  window.listeners.message({ data: { type: "bv-lab-ready" } });
  window.listeners.message({ data: { type: "bv-lab-state", presetIndex: 4, playing: false } });
  out.labBridge = {
    preLoadDirty,
    fullscreenHref,
    postedActions: postedLabMessages.map(m => m.action),
    postSyncDirty: appState.labPresetDirty,
    pauseBtnText: getEl("btnLabPauseResume").textContent
  };

  // 10. Axiom 1 metadata checks + vehicle schematics paths
  out.axiomViolations = [
    ...VIDEO_SCENES.filter(s => (s.series === "phase_atlas") ? (s.axiom1 || !s.comparativeSynthesis) : (!s.axiom1 || s.comparativeSynthesis)),
    ...PLATES_4K.filter(p => (p.series === "phase_atlas") ? (p.axiom1 || !p.comparativeSynthesis) : (!p.axiom1 || p.comparativeSynthesis))
  ].map(x => x.id);

  out.schematicPaths = VEHICLE_SCHEMATICS.flatMap(v => [v.svg, v.yaml]);

  console.log(JSON.stringify(out));
})();
""",
        encoding="utf-8",
    )

    res = subprocess.run([node_bin, str(harness_file)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    report = json.loads(res.stdout.strip())

    # Default initial video is the exact file from the user's request
    assert (
        report["initialVideoSrc"]
        == "experiments/strogatz-dynamics/media/mp4/phase_atlas/daydreamer_16x9.mp4"
    )
    assert report["initialTheme"] == "cosmic"

    # Compare catalog against all 78 actual media files on disk
    disk_mp4s = {
        str(p.relative_to(REPO)).replace("\\", "/")
        for p in MEDIA_DIR.rglob("*.mp4")
    }
    disk_pngs = {
        str(p.relative_to(REPO)).replace("\\", "/")
        for p in MEDIA_DIR.rglob("*.png")
    }
    disk_gifs = {
        str(p.relative_to(REPO)).replace("\\", "/")
        for p in MEDIA_DIR.rglob("*.gif")
    }
    assert len(disk_mp4s) == 30
    assert len(disk_pngs) == 38
    assert len(disk_gifs) == 10

    assert set(report["visitedMp4s"]) == disk_mp4s
    assert len(report["visitedMp4s"]) == 30
    assert set(report["sceneGifs"]) == disk_gifs
    assert len(report["cosmicPlateSrcs"]) == 19
    assert len(report["textbookPlateSrcs"]) == 19
    assert set(report["cosmicPlateSrcs"]).isdisjoint(set(report["textbookPlateSrcs"]))
    assert (set(report["cosmicPlateSrcs"]) | set(report["textbookPlateSrcs"])) == disk_pngs

    assert report["filterCounts"] == {
        "all": 19,
        "phase_portraits": 8,
        "tripanels": 6,
        "phase_atlas": 5,
    }
    assert report["lightboxOk"] is True

    # GitHub /blob/, /raw/, branch-with-slash, share-link & edge-case resolver checks
    assert report["resolveBlobDaydreamer"] == {"sceneIdx": 0, "aspect": "16x9"}
    assert report["resolveBlobWithRawQuery"] == {"sceneIdx": 0, "aspect": "16x9"}
    assert report["resolveRawRefsHeads"] == {"sceneIdx": 0, "aspect": "16x9"}
    assert report["resolveRawUserContent"] == {"sceneIdx": 4, "aspect": "9x16"}
    assert report["resolveSlashBranch"] == {"sceneIdx": 5, "aspect": "1x1"}
    assert report["resolveMonographGif"] == {"sceneIdx": 4, "aspect": None}
    assert report["resolveShareUrl"] == {"sceneIdx": 7, "aspect": "9x16"}
    assert report["resolveBlobShyVertical"] == {"sceneIdx": 4, "aspect": "9x16"}
    assert report["resolveShort"] == {"sceneIdx": 2, "aspect": "1x1"}
    assert report["resolveEmpty"] is None
    assert report["resolveBareExtension"] is None
    assert report["resolveBareAspect"] is None
    assert report["resolveBareRepoUrl"] is None
    assert report["resolveMalformedUri"] is None
    assert report["resolveUnknown"] is None

    assert (
        report["queryLoadedMp4"]
        == "experiments/strogatz-dynamics/media/mp4/v7_wary_moth_9x16.mp4"
    )
    assert report["queryLoadedTheme"] == "textbook"
    assert "--aspect 9:16" in report["queryCliCmd"] and "--theme textbook" in report["queryCliCmd"]
    assert report["fileShareUrl"].startswith(
        "https://dstadulis.github.io/Braitenberg-Vehicles/?scene=v7_wary_moth&aspect=9x16&theme=textbook"
    )
    assert report["unhandledRejected"] is False
    assert report["gifFallbackActivated"] is True
    assert report["gifAutoFallbackClearedOnSceneSelect"] is True
    assert report["gifExitedOnExplicitVideoCtas"] is True
    assert report["gifFallbackRestored"] is True
    assert report["labBridge"]["preLoadDirty"] is True
    assert "preset=v12_daydreamer" in report["labBridge"]["fullscreenHref"]
    assert report["labBridge"]["postedActions"] == ["setPreset", "sync"]
    assert report["labBridge"]["postSyncDirty"] is False
    assert "Resume Lab" in report["labBridge"]["pauseBtnText"]
    assert report["axiomViolations"] == []

    for rel in report["schematicPaths"]:
        p = REPO / rel
        assert p.is_file() and p.stat().st_size > 0, f"Missing schematic or YAML: {rel}"


def test_github_pages_workflow_yaml_and_bundle_completeness(tmp_path: Path) -> None:
    wf = yaml.safe_load(PAGES_WORKFLOW.read_text(encoding="utf-8"))
    # Note: PyYAML 1.1 parses bare `on:` as boolean True unless quoted, check either key
    triggers = wf.get("on") or wf.get(True)
    assert triggers is not None
    assert "main" in triggers["push"]["branches"]
    assert "workflow_dispatch" in triggers

    perms = wf["permissions"]
    assert perms["pages"] == "write"
    assert perms["id-token"] == "write"

    jobs = wf["jobs"]
    assert "build" in jobs and "deploy" in jobs
    assert jobs["deploy"]["needs"] == "build"

    # Execute the exact bundle-assembly & verification steps from pages.yml in tmp_path
    build_steps = {s["name"]: s.get("run", "") for s in jobs["build"]["steps"] if "name" in s}
    assemble_cmd = build_steps["Assemble static site bundle"]
    verify_cmd = build_steps["Verify zero broken static asset paths"]

    # Create lightweight symlink mirror of REPO in tmp_path/work so `cp -R` / symlinks run fast
    work = tmp_path / "work"
    work.mkdir()
    for item in REPO.iterdir():
        if item.name in {".git", ".venv", ".pytest_cache", "__pycache__"}:
            continue
        (work / item.name).symlink_to(item)

    res_assemble = subprocess.run(["bash", "-euo", "pipefail", "-c", assemble_cmd], cwd=work, capture_output=True, text=True)
    assert res_assemble.returncode == 0, res_assemble.stderr

    res_verify = subprocess.run(["bash", "-euo", "pipefail", "-c", verify_cmd], cwd=work, capture_output=True, text=True)
    assert res_verify.returncode == 0, res_verify.stderr
    assert "30 MP4s, 38 PNGs, 10 GIFs" in res_verify.stdout

    # Confirm Verify step genuinely catches a broken interpolated ${MEDIA_ROOT} asset path
    site_index = work / "_site" / "index.html"
    tampered = site_index.read_text(encoding="utf-8").replace(
        "${MEDIA_ROOT}/phase_portraits/v2a_fear_phase.png",
        "${MEDIA_ROOT}/phase_portraits/nonexistent_broken_plate.png",
    )
    site_index.write_text(tampered, encoding="utf-8")
    res_tampered = subprocess.run(["bash", "-euo", "pipefail", "-c", verify_cmd], cwd=work, capture_output=True, text=True)
    assert res_tampered.returncode != 0
    assert "nonexistent_broken_plate.png" in res_tampered.stderr


def test_explorer_html_query_params_and_postmessage_bridge(tmp_path: Path) -> None:
    node_bin = _node_binary()
    if node_bin is None:
        pytest.skip("node binary not installed")

    explorer_html = (REPO / "experiments" / "strogatz-dynamics" / "explorer.html").read_text(encoding="utf-8")
    assert "http://" not in explorer_html and "https://" not in explorer_html
    js_code = re.search(r"<script>([\s\S]*?)</script>", explorer_html).group(1)

    harness = tmp_path / "explorer_bridge_verify.js"
    harness.write_text(
        """
function makeMockCtx() {
  return {
    save() {}, restore() {}, beginPath() {}, closePath() {},
    moveTo() {}, lineTo() {}, stroke() {}, fill() {},
    fillRect() {}, strokeRect() {}, roundRect() {}, arc() {},
    translate() {}, rotate() {}, setLineDash() {},
    fillText() {}, strokeText() {}, drawImage() {},
    createRadialGradient() { return { addColorStop() {} }; },
    createImageData(w, h) { return { data: new Uint8ClampedArray(w * h * 4) }; },
    putImageData() {}
  };
}
const elements = {};
function getEl(id) {
  if (!elements[id]) {
    elements[id] = {
      id, width: 840, height: 840, value: "composite", textContent: "", innerHTML: "",
      style: {}, attrs: {}, listeners: {},
      classList: { toggle() {}, add() {}, remove() {} },
      appendChild() {},
      setAttribute(k, v) { this.attrs[k] = String(v); },
      getAttribute(k) { return this.attrs[k] ?? null; },
      addEventListener(ev, fn) { this.listeners[ev] = fn; },
      getBoundingClientRect() { return { left: 0, top: 0, width: 840, height: 840 }; },
      getContext() { return makeMockCtx(); },
      toDataURL() { return "data:image/png;base64,AA=="; },
      click() { if (this.listeners.click) this.listeners.click(); }
    };
  }
  return elements[id];
}
const docEl = getEl("htmlRoot");
docEl.setAttribute("data-theme", "cosmic");
const document = {
  createElement(tag) { return getEl("dyn_" + tag + "_" + Math.random()); },
  getElementById(id) { return getEl(id); },
  querySelectorAll() { return []; },
  documentElement: docEl
};
const parentMessages = [];
const window = {
  location: { search: "?preset=v12_daydreamer&theme=textbook", hash: "" },
  parent: { postMessage(msg) { parentMessages.push(msg); } },
  listeners: {},
  addEventListener(ev, fn) { this.listeners[ev] = fn; }
};
function requestAnimationFrame() {}
"""
        + js_code
        + """
const bootPreset = PRESETS[state.presetIndex].id;
const bootTheme = state.theme;
const readySent = parentMessages.some(m => m.type === "bv-lab-ready" && m.presetId === "v12_daydreamer" && m.theme === "textbook");

// Exercise cross-origin postMessage control (setPreset, setTheme, togglePlay)
window.listeners.message({ data: { type: "bv-lab-control", action: "setPreset", presetIdx: 3, theme: "cosmic" } });
const afterMsgPreset = PRESETS[state.presetIndex].id;
const afterMsgTheme = state.theme;

const playingBefore = state.playing;
window.listeners.message({ data: { type: "bv-lab-control", action: "togglePlay" } });
const playingAfter = state.playing;

console.log(JSON.stringify({ bootPreset, bootTheme, readySent, afterMsgPreset, afterMsgTheme, playingBefore, playingAfter }));
""",
        encoding="utf-8",
    )

    res = subprocess.run([node_bin, str(harness)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout.strip())
    assert out == {
        "bootPreset": "v12_daydreamer",
        "bootTheme": "textbook",
        "readySent": True,
        "afterMsgPreset": "v7_wary_moth",
        "afterMsgTheme": "cosmic",
        "playingBefore": True,
        "playingAfter": False,
    }
