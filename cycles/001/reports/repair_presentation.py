"""Repair copied-document links and expose all frozen settings without fitting.

Run after controller rendering. Empirical tables stay untouched. The interpretation
is the persistent source of the method supplement; copies resolve links from its
original directory. Pandoc embeds local images and makes repository links usable
in the standalone PDF. https://pandoc.org/MANUAL.html (resource paths, PDF output).
"""
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
REPOSITORY = 'https://github.com/BurnyCoder/eureka-digits-sol/blob/main/'


def portable_links(markdown, origin):
    """Resolve relative document targets before copying to the site or PDF."""
    def replace(match):
        label, target = match[1], match[2]
        if re.match(r'https?://|mailto:|#', target):
            return match[0]
        local, separator, fragment = target.partition('#')
        destination = (origin / local).resolve()
        assert destination.exists(), destination
        relative = destination.relative_to(PROJECT).as_posix()
        base = (REPOSITORY.replace('github.com/', 'raw.githubusercontent.com/').replace('/blob/', '/')
                if label.startswith('!') else REPOSITORY.replace('/blob/', '/tree/') if destination.is_dir() else REPOSITORY)
        return label + '(' + base + relative + (separator + fragment if separator else '') + ')'
    return re.sub(r'(!?\[[^\]]*\])\(([^)]+)\)', replace, markdown)


def main():
    """Update derived presentation from preserved scientific files, then compile."""
    protocol = json.loads((ROOT / 'research/protocol.json').read_text(encoding='utf-8'))
    interpretation = ROOT / 'reports/interpretation.md'
    text = interpretation.read_text(encoding='utf-8')
    text = portable_links(text, interpretation.parent)
    # Long prose cells can overflow a PDF page. Preserve their content as prose.
    prediction_table = re.search(r'\| Earlier prediction or alternative \|[^\n]*\n\| ---[^\n]*\n((?:\|[^\n]*\n)+)', text)
    if prediction_table:
        paragraphs = []
        for row in prediction_table[1].strip().splitlines():
            prediction, observation, explanation = [part.strip() for part in row.strip('|').split('|')]
            paragraphs.append(f'**Prediction or alternative:** {prediction}\n\n**Observation:** {observation}\n\n**Interpretation:** {explanation}')
        text = text[:prediction_table.start()] + '\n\n'.join(paragraphs) + '\n' + text[prediction_table.end():]
    marker = '<!-- BEGIN FROZEN SETTINGS -->'
    if marker not in text:
        keys = ('architecture', 'activation', 'optimizer', 'learning_rate', 'batch_size',
                'epochs', 'seeds', 'split_seed', 'logistic_C',
                'logistic_max_iter', 'bootstrap_seed', 'bootstrap_resamples')
        settings = '\n\n' + marker + '\n\n### Complete frozen settings\n\n'
        settings += 'Generated directly from the registered protocol, without outcome-based selection.\n\n'
        settings += '| Setting | Registered value |\n| --- | --- |\n'
        settings += '\n'.join(f'| `{key}` | `{json.dumps(protocol[key])}` |' for key in keys)
        settings += '\n' + '\n'.join(f'| `weight_decay.{method}` | `{value}` |' for method, value in protocol['weight_decay'].items())
        settings += '\n\n<!-- END FROZEN SETTINGS -->\n\n'
        text = text.replace('## Predictions, observations and negative findings', settings + '## Predictions, observations and negative findings')
    interpretation.write_text(text, encoding='utf-8')
    paper_path = ROOT / 'reports/paper.md'
    paper = paper_path.read_text(encoding='utf-8')
    start = paper.index('# Interpretation of the fixed digits replication')
    end = paper.index('\n\n# Limitations', start)
    paper = paper[:start] + text.rstrip() + paper[end:]
    paper = paper.replace('See `docs/reproduction.md` for execution commands and environment details.',
                          f'See [the reproduction guide]({REPOSITORY}cycles/001/docs/reproduction.md) for execution commands and environment details.')
    # Link paper images to portable repository targets in the site Markdown only;
    # local paths must remain in the PDF input so Pandoc can embed actual pixels.
    paper_path.write_text(paper, encoding='utf-8')
    (PROJECT / 'site/paper.md').write_text(portable_links(paper, paper_path.parent), encoding='utf-8')
    subprocess.run(['pandoc', str(paper_path), '--from', 'markdown-implicit_figures', '--resource-path', str(paper_path.parent),
                    '--pdf-engine=pdflatex', '-o', str(ROOT / 'reports/paper.pdf')], check=True, cwd=ROOT)
    shutil.copyfile(ROOT / 'reports/paper.pdf', PROJECT / 'site/paper.pdf')
    # Reapply the small UI provenance repair after a controller regenerates HTML.
    site_path = PROJECT / 'site/index.html'
    original = '`Example ${first.example_ids[i]} · true ${truth} · predicted ${first.y_pred[i]}`'
    replacement = '`${first.id} · ${first.split} · example ${first.example_ids[i]} · true ${truth} · predicted ${first.y_pred[i]}`'
    site_path.write_text(site_path.read_text(encoding='utf-8').replace(original, replacement), encoding='utf-8')


if __name__ == '__main__':
    main()
