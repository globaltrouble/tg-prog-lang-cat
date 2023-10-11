## ToDo

Collect dataset for those classes:

 - [X] TGLANG_LANGUAGE_1S_ENTERPRISE
 - [X] TGLANG_LANGUAGE_APEX
 - [ ] TGLANG_LANGUAGE_DELPHI (can download files from here https://github.com/danieleteti/delphimvcframework/tree/master)
 - [ ] TGLANG_LANGUAGE_FIFT (very rare language https://github.com/Piterden/TON-docs/blob/master/Fift.%20A%20Brief%20Introduction.md)
 - [ ] TGLANG_LANGUAGE_FUNC (does it mean functional languages like Haskell)
 - [X] TGLANG_LANGUAGE_GRADLE
 - [X] TGLANG_LANGUAGE_HACK
 - [ ] TGLANG_LANGUAGE_ICON (can download snippets from here https://github.com/gtownsend/icon)
 - [ ] TGLANG_LANGUAGE_KEYMAN (very rare language fot keyboards, maybe can download something from here https://github.com/keymanapp/keyman)
 - [X] TGLANG_LANGUAGE_LOGO
 - [X] TGLANG_LANGUAGE_OPENEDGE_ABL
 - [X] TGLANG_LANGUAGE_REGEX
 - [X] TGLANG_LANGUAGE_TL


Fix issue with overlapping classes:

 - [ ] TGLANG_LANGUAGE_PL_SQL (overlapping with TGLANG_LANGUAGE_SQL)
 - [ ] TGLANG_LANGUAGE_VBSCRIPT (overlapping with TGLANG_LANGUAGE_VISUAL_BASIC)
 - [ ] TGLANG_LANGUAGE_LISP (overlapping with TGLANG_LANGUAGE_COMMON_LISP)


Data issues we spotted to be cleaned in the dataset:
- [ ] Copyrights / Licences (original dataset)
- [ ] Comments removal???
- [ ] Natural text in files, no code (no special symbols in the file e.g.) - CSV files? - case tabulations
- [ ] Cases like "=========HERE'S THE MIGHTY CODE============") - replace them? (10 consecutive symbols -> remove - try it)


Modeling TODOs:
- [X] Resolve the issue with the low accuracy without shuffling
- [X] Migrate the solution to use it on the dataset from Vadim
- [ ] Write the script for per-class evaluation
- [ ] Try the approach with removing the comments from the source code and evaluate




