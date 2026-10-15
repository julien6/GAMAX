# Extensions du contrôle de flux — 12–13 octobre 2026

Le compilateur traduit désormais les chaînes `else if`, les boucles `loop times: N` sans variable, les boucles `loop i from: A to: B step: S` et les appels d'actions avec arguments nommés sous forme de facettes. La borne finale des boucles de plage est inclusive ; un pas négatif est accepté. Le corps peut appeler des actions, déclarer des variables locales et tirer des nombres aléatoires. Les variables locales et la variable de plage ne sortent pas de leur bloc.

Ces boucles sont développées statiquement en transitions JAX ordonnées : les mises à jour et les tirages d'une itération précèdent ceux de la suivante. Les conditions des branches sont figées avant leurs écritures et les branches inactives ne consomment aucun tirage. Les arguments nommés sont évalués dans **l'ordre des paramètres déclarés**, établi par une comparaison native qui inverse volontairement deux arguments contenant `rnd`.

## Contrat et limites

Les bornes et le pas doivent être des entiers littéraux int32. Le pas ne peut pas être nul. Une boucle est limitée à 1 024 itérations et l'expansion d'une action, d'un reflex ou d'un init à 4 096 statements. Les corps de boucles à zéro itération restent vérifiés pour les noms et les types. Les boucles à bornes calculées, les parcours de collections, `break` et `continue` restent rejetés. Les arguments nommés doivent fournir exactement les paramètres de l'action, sans doublon ; les appels concernent les actions du même scope. Cela ne prend pas en charge les facettes de `move`, `wander` ou `goto`.

## Validation

Le [modèle GAML](../examples/control_flow/main.gaml) associe boucles imbriquées, pas négatif, boucle vide, boucle répétée sans variable, branches `else if`, actions et arguments aléatoires nommés écrits en ordre inverse. GAMA et GAMAX exécutent exactement la même source. La campagne conserve dix seeds positifs, 100 cycles et chaque état, initialisation comprise. Elle rejoue aussi les trois profils précédents, soit 40 configurations. Une campagne finale ajoute dix seeds d’initialisation aléatoire, pour un total de 50 configurations et 178 770 valeurs comparées. Tous les cas passent ; les 47 470 valeurs des deux nouveaux profils sont exactement identiques. L’erreur flottante maximale de la campagne, issue du mouvement stochastique déjà couvert, est de 1,734 × 10⁻¹² sous les tolérances préétablies. Les cinq contrôles de divergence sont détectés. Pour le premier seed du nouveau profil, retirer une itération extérieure doit faire diverger les résultats.

Les 86 tests Python passent. Ils couvrent notamment les diagnostics avant création des fichiers générés, les variables de boucle hors portée, le shadowing, les bornes invalides, les noms d'arguments invalides et le plafonnement des expansions imbriquées.

La [campagne initiale en échec](../results/compiler-control-flow-final-2026-10-12/summary.json) est conservée : ses dix nouveaux cas détectent l'ordre incorrect d'évaluation des arguments nommés, alors que les trente anciens cas passent. La [campagne corrigée du contrôle de flux](../results/compiler-control-flow-corrected-final-2026-10-12/summary.json) contient les traces, contrôles, protocoles et snapshots des implémentations. Les tolérances n'ont pas été modifiées pour corriger l'écart.

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/evaluate_compiler_extensions.py \
  --cases control_flow --steps 100 --seeds 1 2 3 4 5 6 7 8 9 10 \
  -o results/local/new-control-flow
```

Omettre `--cases` pour rejouer les cinq profils. Utiliser un répertoire neuf.


## Initialisation aléatoire des espèces

Le [second modèle](../examples/random_initialization/main.gaml) utilise `rnd` dans les attributs int/float, puis des points et branches `flip` dans les blocs `init`, avant de poursuivre les tirages dans les reflex. Le couple `rng <- "java"; seed <- N;` doit précéder toute création dans le global init. GAMAX initialise d’abord les attributs de chaque agent dans l’ordre de création, puis exécute les blocs init individuels. Le module conserve l’état RNG obtenu pour les transitions suivantes ; `reset()` reproduit les mêmes tirages. Ce profil exclut les espèces à skills, les tirages dans les attributs globaux et les expressions de collection. Le couple RNG/seed final reste disponible pour le profil runtime précédent.

Une première sonde, [conservée en échec](../results/random-initialization-probe-2026-10-12/comparison.json), intercalait à tort les attributs et le bloc init de chaque agent. La correction reproduit les deux phases natives sans ajuster les tolérances. La [campagne finale](../results/compiler-control-flow-initialization-final-2026-10-12/summary.json) valide les cinq profils ensemble. L’[audit de compatibilité](../results/compiler-control-flow-initialization-final-2026-10-12/legacy_generated_code_audit.json) confirme que les 131 modules régénérés des archives antérieures sont identiques byte pour byte ; cet audit ne constitue pas une nouvelle exécution native de ces archives.

## Couverture officielle restante

L'[audit actuel](../benchmarks/corpus-initialization-final-2026-10-13/coverage.json) rejette toujours les six grands modèles officiels retenus, et quatre autres familles attendent une source officielle sélectionnée. Cette extension ne démontre donc ni leur compilation complète ni un gain de vitesse supplémentaire. Les blocages comprennent encore l’initialisation implicite des skills et le RNG Mersenne par défaut, l'héritage/overrides, les références d'agents, les collections modifiables, les corps de `create`, les naissances/morts, les grilles et requêtes spatiales, `wander`/`goto`, les FSM/diffusions et les expérimentations graphiques avec effets. Les sources officielles n'ont pas été simplifiées pour produire les résultats ci-dessus.
