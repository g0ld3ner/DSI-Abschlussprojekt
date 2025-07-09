#!/usr/bin/env python3
import ast
import pathlib
import subprocess


def find_imports(path):
    imports = set()
    for file in pathlib.Path(path).rglob('*.py'):
        try:
            source = file.read_text()
            tree = ast.parse(source, filename=str(file))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split('.')[0])
    return sorted(imports)


if __name__ == '__main__':
    # Pfad zum Quellcode-Verzeichnis
    src_path = pathlib.Path(__file__).parent.parent / 'src'
    packages = find_imports(src_path)

    # requirements.txt schreiben
    req_file = pathlib.Path(__file__).parent.parent / 'requirements.txt'
    with req_file.open('w') as f:
        for pkg in packages:
            try:
                # Version abrufen
                result = subprocess.run(['pip', 'show', pkg], capture_output=True, text=True, check=True)
                for line in result.stdout.splitlines():
                    if line.startswith('Version:'):
                        version = line.split(':', 1)[1].strip()
                        f.write(f"{pkg}=={version}\n")
                        break
            except subprocess.CalledProcessError:
                # Paket ohne Version hinzufügen
                f.write(f"{pkg}\n")

    print('requirements.txt wurde aktualisiert.')
