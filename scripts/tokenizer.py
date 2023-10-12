from typing import List
import re

WORDS_AND_TOKENS_REGEX = r"\b(\w+)\b|([.,;:\\\/{}\[\]\|!\"#\$%&\'\(\)\*\+-\<\=\>\?@\^\`\~)])"


def tokenize(text: str) -> List[str]:
    
    """
    Tokenizes a string by words and special symbols that are contained in most programming languages.
    :
    
    """

    results = re.findall(WORDS_AND_TOKENS_REGEX, text.lower())
    entries = [entry for tpl in results for entry in tpl if entry]

    return entries


assert tokenize("something . just . like - this \ happens / a | lot { ago } here ' s [ the ] result ; you : expect ! sdfsdfasdf ? asfdasdfasd # asdfasdfasdf $ adsfasdfad % adfasdfad &  '  (  )  *  +  -  <  =  >  ?  @  ^  `  ~ ") == \
    ['something', '.', 'just', '.', 'like', '-', 'this', '\\', 'happens', '/', 'a', '|', 'lot', '{', 'ago', '}', 'here', "'", 's', '[', 'the', ']', 'result', ';', 'you', ':', 'expect', '!', 'sdfsdfasdf', '?', 'asfdasdfasd', '#', 'asdfasdfasdf', '$', 'adsfasdfad', '%', 'adfasdfad', '&', "'", '(', ')', '*', '+', '-', '<', '=', '>', '?', '@', '^', '`', '~']