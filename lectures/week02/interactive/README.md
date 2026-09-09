# Lecture 02 interactive player

This is a self-contained browser version of the Lecture 02 interactive exercises.
Python is used only to serve static files; all calculations run locally in your
browser. After downloading the repository, the player does not require internet
access.

## Start the player

From the repository root, run:

```sh
python3 -m http.server 2718 --bind 127.0.0.1 --directory lectures/week02/interactive/app
```

Then open <http://127.0.0.1:2718> in a recent Chrome, Edge, Firefox, or Safari
browser. The first load can take a few seconds because the local Python/WebAssembly
runtime is about 60 MB.

On Windows, use `py` instead of `python3` if that is how Python is installed:

```powershell
py -m http.server 2718 --bind 127.0.0.1 --directory lectures/week02/interactive/app
```

Press `Ctrl-C` in the terminal to stop the server. If port 2718 is already in
use, replace `2718` in both the command and URL with another port such as `8000`.

Do not open `app/index.html` directly with a `file://` URL. Browser workers require
a local HTTP origin.

## Controls

| Key or control | Action |
|---|---|
| Left / Right | Previous or next scene |
| Space / Reveal | Reveal the next core explanation layer |
| Down / Deepen | Open optional detail |
| Up / Core | Return to the core path |
| A / Answer | Show or hide the transfer answer |
| R / Reset | Reset the current scene's mathematical values |
| Scene menu | Jump directly to one of the 18 scenes |

When typing in an editable field, keyboard navigation is paused. A complete scene
link includes its scene and layer, for example:
<http://127.0.0.1:2718/?scene=L02-S09&layer=explain&depth=0&answer=0>.

## If the player does not load

1. Confirm that the terminal still shows the local server running.
2. Reload <http://127.0.0.1:2718> rather than opening the HTML file directly.
3. Try another browser or another port.
4. Continue with [the Lecture 02 PDF](../lecture02.pdf); it contains the complete
   core teaching path and static versions of every interactive scene.
