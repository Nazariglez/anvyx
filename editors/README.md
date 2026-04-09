# Editor Support

This folder contains editor support for Anvyx `.anv` files.

Right now there are two options:

- `vscode/` for VS Code
- `nvim/` for Neovim with Tree-sitter

The goal of this README is simple: help you get syntax highlighting working.

## VS Code

The easiest way to try the extension is to open your editor with the extension loaded from this repo.

```bash
code --extensionDevelopmentPath="$(pwd)/editors/vscode"
```

If you want to keep it installed locally, symlink the folder into your extensions directory and restart the editor:

```bash
ln -s "$(pwd)/editors/vscode" ~/.vscode/extensions/anvyx
```

After that, open any `.anv` file. Highlighting should start automatically.

## Neovim

The Neovim support uses Tree-sitter.

You need:

- Neovim 0.9 or newer
- `nvim-treesitter`
- Node.js
- a C compiler

First, build the parser from the repo root:

```bash
cd editors/nvim
npm install
npx tree-sitter generate
```

Then add this to your `init.lua`:

```lua
vim.filetype.add({ extension = { anv = "anvyx" } })

local parser_config = require("nvim-treesitter.parsers").get_parser_configs()
parser_config.anvyx = {
  install_info = {
    url = "/path/to/this/repo/editors/nvim",
    files = { "src/parser.c" },
  },
  filetype = "anvyx",
}
```

Replace `/path/to/this/repo` with the real path to your checkout.

Then install the parser inside Neovim:

```vim
:TSInstall anvyx
```

Next, make the highlight queries available:

```bash
mkdir -p ~/.local/share/nvim/queries/anvyx
ln -sf /path/to/this/repo/editors/nvim/queries/highlights.scm \
  ~/.local/share/nvim/queries/anvyx/highlights.scm
```

Make sure Tree-sitter highlighting is enabled in your Neovim config:

```lua
require("nvim-treesitter.configs").setup({
  highlight = { enable = true },
})
```

After that, open any `.anv` file and highlighting should work.
