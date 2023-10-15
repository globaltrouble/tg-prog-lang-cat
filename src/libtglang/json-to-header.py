import sys
import json

data = []
for l in sys.stdin:
    data.append(l)
data = '\n'.join(data)

print("""
#ifndef itfidf_mapping_H
#define itfidf_mapping_H

#include <utility>
#include <cstdint>

const std::pair<const char *, std::pair<float, int64_t>> raw_itfidf_mapping[] = {
""")
for ind, (k, v) in enumerate(json.loads(data).items()):
    f, i = v
    if ind > 0:
        print(",")
    print('{"%s", {%s, %s}}' % (k.replace("\\", "\\\\").replace('"', r'\"'), f, i), end="")
    # if i > 1000:
    #     break       # TODO: remove!
print("""
};
#endif // itfidf_mapping_H
# """)
