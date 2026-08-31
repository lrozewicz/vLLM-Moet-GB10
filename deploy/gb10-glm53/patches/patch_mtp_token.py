#!/usr/bin/env python3
"""LATKA 4: MTP na runnerze V1 dla GLM-5.3-Flash (blad nazwy atrybutu).

PROBLEM. llm_base_proposer.load_model ma lancuch if/elif/else z nazwami
atrybutu tokenu obrazu per rodzina modeli. Galaz `else` zaklada
`target_model.config.image_token_index`, a Glm5NextConfig wystawia
`image_token_id`. Efekt:

  AttributeError: 'Glm5NextConfig' object has no attribute 'image_token_index'.
  Did you mean: 'image_token_id'?

Dlatego MTP dziala na runnerze V2 (inna sciezka proposera), a wywala sie na
V1 - a V1 jest nam potrzebny, bo tam siedza hooki warstwy FP4. To jedyna
rzecz, ktora blokuje polaczenie JAKOSCI (FP4) z SZYBKOSCIA (MTP).

LATKA. Zamiast dokladac galez per model, probuj po kolei nazw, ktore
faktycznie wystepuja: image_token_index -> image_token_id ->
vision_config.image_token_id. Pierwsza znaleziona wygrywa.
"""
import sys

path = sys.argv[1]
src = open(path, encoding="utf-8").read()
if "[PATCH-MTP-TOKEN]" in src:
    print("juz zalatane"); sys.exit(0)

OLD = """            else:
                self.model.config.image_token_index = (
                    target_model.config.image_token_index
                )"""
NEW = """            else:
                # [PATCH-MTP-TOKEN] Nazwa atrybutu roznii sie miedzy rodzinami:
                # Glm5NextConfig ma `image_token_id`, nie `image_token_index`.
                # Bierzemy pierwsza, ktora faktycznie istnieje.
                _tok = None
                for _src, _name in (
                    (target_model.config, "image_token_index"),
                    (target_model.config, "image_token_id"),
                    (getattr(target_model.config, "vision_config", None),
                     "image_token_id"),
                ):
                    if _src is not None and hasattr(_src, _name):
                        _tok = getattr(_src, _name)
                        break
                if _tok is None:
                    raise AttributeError(
                        "nie znaleziono id tokenu obrazu w konfiguracji celu "
                        f"({type(target_model.config).__name__}); sprawdzone: "
                        "image_token_index, image_token_id, "
                        "vision_config.image_token_id")
                self.model.config.image_token_index = _tok"""

if OLD not in src:
    print("NIE ZNALEZIONO kotwicy", file=sys.stderr); sys.exit(1)
open(path, "w", encoding="utf-8").write(src.replace(OLD, NEW, 1))
import ast; ast.parse(open(path, encoding="utf-8").read())
print("zalatane, skladnia OK")
