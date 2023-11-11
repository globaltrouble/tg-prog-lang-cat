## ToDo

Collect dataset for those classes:

 - [X] TGLANG_LANGUAGE_1S_ENTERPRISE
 - [X] TGLANG_LANGUAGE_APEX
 - [ ] TGLANG_LANGUAGE_DELPHI (can download files from here https://github.com/danieleteti/delphimvcframework/tree/master) <-> or is it pascal
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



TODOs 11.11:

Dataset:
- [X] Prepare and send the dataset in json format (Anton M.)
- [ ] Add 2 remaining classes (FunC and TL) (Can FunC be scrapped from [here](https://github.com/search?q=path%3A*.fc&type=code&p=2) or [here](https://docs.ton.org/develop/smart-contracts/examples#ton-smart-challenge-4))
- [ ] "Label" current TG dataset (OTHER vs. CODE, label code part) with our classifier.
- [ ] Multiple languages in OTHER (currently a lot of russian texts there), maybe augment using data from [here](https://bitbucket.org/globaltrouble/tg-cat/src/master/contest-static/dc0202-input.txt)
- [ ] Add our dataset from BigCode The Stack (with multiple programming languages labeled)



Modeling:
- [ ] Create a Fasttext baseline model for case OTHER vs. CODE
- [ ] Create a Fasttext baseline model for guessing the programming language


Inference:

- [ ] Update the code to load the model in compilation time
- [ ] Update builder on repo for submission


General Modeling Guidelines:
- Create 2 classifiers - one for code vs. other, one for specific programming language
- Use ML libraries for C++ which have Python bindings


Nice-to-have things:
- Explore other solutions for inspiration
- Try things like "guesslang tools"





