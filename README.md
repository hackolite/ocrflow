
# OCRFLOW

OCRFLOW est une plateforme polyvalente conçue pour le benchmarking des pipelines OCR (Optical Character Recognition) et des plateformes de detection d'objets. Elle offre une solution complète on-premise, installable sur un serveur, conçue pour évaluer les performances et la précision des systèmes de reconnaissance de caractères ainsi que des systèmes de détection d'objets, le tout à l'aide d'une architecture flexible et évolutive.


## Architecture OCRFLOW



<picture>
 <source media="(prefers-color-scheme: dark)" srcset="">
 <source media="(prefers-color-scheme: light)" srcset="https://i.postimg.cc/3NmbvXQN/OCRFLOW.png">
 <img alt="YOUR-ALT-TEXT" src="https://i.postimg.cc/3NmbvXQN/OCRFLOW.png">
</picture>


# Philosophie

OCRFLOW est un assemblage de solutions agencées pour gérer le benchmark, l'inférence, l'entrainement de modéles de computervision. A terme la porte d'entrée de la plateforme sera l'API.

### Inférence
Pour le moment l'inférence temps-réel n'est pas prévue, le scénario d'inférence est asynchrone et fonctionne par passage d'url de l'image à l'API, en choisissant le modéle adapté, stocké au préalable dans la solution de stockage objet MINIO.
L'image est processée par un consumer inscrit sur une queue rabbitmq dédiée. 


### Benchmark
Le benchmark peut être réalisé par passage d'url ou par passage de bucket MINIO. 
Les métriques sont déclarées dans l'API (Accuracy, Recall, FPS, Mémoire), ainsi que le modéle, le consumer doit renvoyer ses spécifications pour une analyse équitable. 


### Training
Le training se fait par déclaration de bucket, avec déclaration des métriques triggers accuracy, nombre d'épochs etc ...
Le modéle issu de l'entrainement est ensuite stocké dans MINIO. 


# Déploiement

To deploy this project run

```bash
  npm run deploy
```

## API Reference

#### Get all items

```http
  GET /api/items
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `api_key` | `string` | **Required**. Your API key |

#### Get item

```http
  GET /api/items/${id}
```

| Parameter | Type     | Description                       |
| :-------- | :------- | :-------------------------------- |
| `id`      | `string` | **Required**. Id of item to fetch |

#### add(num1, num2)

Takes two numbers and returns the sum.


## Appendix

Any additional information goes here


## Authors

- [@octokatherine](https://www.github.com/octokatherine)


## Badges

Add badges from somewhere like: [shields.io](https://shields.io/)

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)

## Color Reference

| Color             | Hex                                                                |
| ----------------- | ------------------------------------------------------------------ |
| Example Color | ![#0a192f](https://via.placeholder.com/10/0a192f?text=+) #0a192f |
| Example Color | ![#f8f8f8](https://via.placeholder.com/10/f8f8f8?text=+) #f8f8f8 |
| Example Color | ![#00b48a](https://via.placeholder.com/10/00b48a?text=+) #00b48a |
| Example Color | ![#00d1a0](https://via.placeholder.com/10/00b48a?text=+) #00d1a0 |


## Contributing

Contributions are always welcome!

See `contributing.md` for ways to get started.

Please adhere to this project's `code of conduct`.


## Demo

Insert gif or link to demo


## Deployment

To deploy this project run

```bash
  npm run deploy
```


## Documentation

[Documentation](https://linktodocumentation)


## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`API_KEY`

`ANOTHER_API_KEY`


## Installation

Install my-project with npm

```bash
  npm install my-project
  cd my-project
```
    
## Lessons Learned

What did you learn while building this project? What challenges did you face and how did you overcome them?


## License

[MIT](https://choosealicense.com/licenses/mit/)


## Optimizations

What optimizations did you make in your code? E.g. refactors, performance improvements, accessibility


## Run Locally

Clone the project

```bash
  git clone https://link-to-project
```

Go to the project directory

```bash
  cd my-project
```

Install dependencies

```bash
  npm install
```

Start the server

```bash
  npm run start
```


## Roadmap

- Additional browser support

- Add more integrations


## Tech Stack

**Client:** React, Redux, TailwindCSS

**Server:** Node, Express


## Usage/Examples

```javascript
import Component from 'my-project'

function App() {
  return <Component />
}
```


## Used By

This project is used by the following companies:

- Xretail
