import sys
if len(sys.argv) == 1:
    wiki_dir = './'
else:
    wiki_dir = '{}/'.format(str(sys.argv[1]))

module = 'plotpal'

raw_file  = 'full_{}.md'.format(module)
full_file = open(raw_file, 'r')

current_section = None
sections = dict()

#Read and split up file
for line in full_file:
    if "## plotpal" in line:
        prev_section = current_section
        current_section = line.split("## ")[-1].replace("\\", "").replace("\n", "")
        sections[current_section] = []
        if prev_section is not None:
            sections[current_section].append(sections[prev_section][-1])
            sections[prev_section].pop(-1)
        line = line.replace("##", "#")
    sections[current_section].append(line)
full_file.close()

for sct, lines in sections.items():
    if sct == 'plotpal':
        filename = 'Home'
    else:
        filename = sct

    write_file = open("{}/{}.md".format(wiki_dir, filename), 'w')
    for line in lines:
        write_file.write(line)
    write_file.close()
            
