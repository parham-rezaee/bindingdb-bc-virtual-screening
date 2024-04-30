from glob import glob
import os


SEGMENTS_DIR='E:/Repository/Videos/Classes/Database/Project/segments'
OUTPUT_DIR='./extracted'
ID_FILES_DIR='./targets'

target_molecules_map = dict()
remaining_mols = dict()

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

for id_file in glob(os.path.join(ID_FILES_DIR, '*.csv')):
    with open(id_file) as f:
        target = os.path.basename(id_file).split('.')[0]
        target_molecules_map[target] = set()
        for line in f.readlines():
            line = line.strip()
            if line.startswith('mol_id') or line=='':
                continue

            molid = line.split(',')[0]
            target_molecules_map[target].add(molid)

        remaining_mols[target] = len(target_molecules_map[target])
        target_outdir = os.path.join(OUTPUT_DIR, target)
        if not os.path.exists(target_outdir):
            os.mkdir(target_outdir)

for segment_file in glob(os.path.join(SEGMENTS_DIR, '*.sdf')):
    print(f"PROCESSING SEGMENT FILE: {segment_file}")
    print(remaining_mols.items())
    buffer = []
    molcount = 0
    f_in = open(segment_file, "r", encoding="utf-8")

    while True:
        line = f_in.readline()
        if line == '':
            print('')
            break

        if not line.startswith("$$$$"):
            buffer.append(line)
            continue

        molcount += 1
        molid = buffer[0].strip()
        uri_safe_molid = molid.replace('::', '__')
        molecule = ''.join(buffer)
        for target in target_molecules_map.keys():
            if molid not in target_molecules_map[target]:
                continue

            try:
                f_out = open(os.path.join(OUTPUT_DIR, target, uri_safe_molid + '.sdf'), 'wt', encoding='utf-8')
                f_out.write(molecule)
                f_out.close()
            except Exception as e:
                print(e)
                print("====== MOLID: " + molid)
                print("====== TARGET: " + target)

            remaining_mols[target] -= 1

        buffer = []
        if molcount % 1000 == 0:
            print('.', end='')

    f_in.close()
