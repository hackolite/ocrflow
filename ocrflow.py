import argparse

def deploy_init(args):
    print("Initialisation du déploiement avec les options suivantes :")
    print("- Nom du projet :", args.project_name)
    print("- Version :", args.version)
    print("- Environnement de déploiement :", args.environment)
    print("- Serveur cible :", args.server)

def main():
    parser = argparse.ArgumentParser(description="Script de déploiement")
    subparsers = parser.add_subparsers(title="sous-commandes", dest="subcommand")

    # Sous-commande pour initialiser un déploiement
    parser_init = subparsers.add_parser("init", help="Initialise un déploiement")
    parser_init.add_argument("project_name", help="Nom du projet")
    parser_init.add_argument("--version", help="Version du projet", default="1.0")
    parser_init.add_argument("--environment", help="Environnement de déploiement", default="production")
    parser_init.add_argument("--server", help="Serveur cible", required=True)

    args = parser.parse_args()

    if args.subcommand == "init":
        deploy_init(args)

if __name__ == "__main__":
    main()

