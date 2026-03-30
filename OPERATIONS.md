# Numerai - Guide d'exploitation

## Architecture

Les **predictions quotidiennes** tournent sur **AWS** (ECS Fargate), declenchees automatiquement par les webhooks Numerai. Rien ne tourne en local au quotidien.

Le **training** se fait en local sur ta machine (GPU RTX 4060), uniquement quand tu decides de mettre a jour les modeles.

## Modeles deployes

| Modele   | Tournament    | Node AWS         | Pickle         |
|----------|---------------|------------------|----------------|
| TGRV2    | Classic (t8)  | numerai-tgrv2    | tgrv2.pkl      |
| TGR_SIG  | Signals (t11) | signals-tgr_sig  | signals.pkl    |
| TGR_CRY  | Crypto (t12)  | crypto-tgr_cry   | salok1.pkl     |

## Fonctionnement quotidien

**Rien a faire.** Chaque jour, Numerai envoie un webhook vers tes Lambda AWS, qui lancent les containers ECS. Les containers chargent les pickles, telechargent les donnees live, generent les predictions et les soumettent automatiquement.

Tu peux eteindre ton PC, fermer Docker — les submissions continueront.

## Re-training (quand tu veux mettre a jour les modeles)

Les modeles restent valides pendant des semaines. Un re-training toutes les 1-2 semaines suffit, ou quand Numerai met a jour les datasets.

### Procedure

1. **Lancer Docker Desktop** (necessaire uniquement pour le deploy)
2. **Re-entrainer et deployer** :
   ```powershell
   pwsh -File daily_retrain_deploy.ps1
   ```
   Ce script :
   - Entraine Classic, Signals et Crypto (GPU)
   - Construit les pickles
   - Deploie les containers sur AWS via Docker + ECR

3. **Verifier que tout fonctionne** :
   ```powershell
   pwsh -File health_check.ps1
   ```

4. **Fermer Docker Desktop** — plus besoin jusqu'au prochain re-training.

### Deploy seul (sans re-training)

Si tu as deja des pickles a jour et veux juste redeployer :
```powershell
$n = "C:\Users\nicol\AppData\Roaming\Python\Python313\Scripts\numerai.exe"
& $n node -m tgrv2 -t 8 deploy -v
& $n node -m tgr_sig -t 11 deploy -v
& $n node -m tgr_cry -t 12 deploy -v
```

### Health check seul
```powershell
pwsh -File health_check.ps1
```

## Configuration

- **API Keys Numerai** : `~/.numerai/.keys` (regenerer sur numer.ai si expirees)
- **API Keys locales** : `keys_local.ps1` (git-ignore)
- **Nodes AWS** : `~/.numerai/nodes.json`
- **Training configs** : `numerai-project/config/`
- **Hyperparametres LightGBM** : `numerai-project/config/program_input_params.yaml`

## Depannage

| Symptome | Cause | Solution |
|----------|-------|----------|
| `Your session is invalid or has expired` | API keys expirees | Regenerer sur numer.ai, mettre a jour `~/.numerai/.keys` |
| `failed to connect to docker API` | Docker Desktop eteint | Lancer Docker Desktop |
| `numerai not found` | CLI pas dans le PATH | Utiliser le chemin complet (voir deploy seul) |
| Submissions manquees sur le dashboard | Webhook/node en erreur | Lancer `health_check.ps1` |
