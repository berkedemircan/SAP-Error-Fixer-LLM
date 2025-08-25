# SAP Error Fixer LLM 🚀

This project is a Large Language Model (LLM) based assistant designed to automatically analyze SAP error logs and generate solution recommendations. The model is built on NanoGPT and fine-tuned with domain-specific data, enabling it to provide solutions in both Turkish and English. The system integrates a FastAPI backend with a React-based frontend, allowing users to interact with the model through a simple web interface.  

Key features of the project include multilingual error resolution (Turkish & English), automated log processing, and domain-specific fine-tuning on approximately 1,500 SAP error logs. During training, the model achieved over 90% token accuracy and an F1 score above 0.9, demonstrating strong performance on text-to-solution mapping tasks.  

The technology stack includes Python for data processing, NanoGPT for the LLM architecture and fine-tuning, FastAPI for backend integration, and React.js for the user interface. Git LFS is used to manage large files such as trained model checkpoints.  

The project structure is organized into three main components: **LLMProject** for data preprocessing and augmentation, **nanoGPT** for model training and configuration, and **sap-chatbot-backend** for serving the model via FastAPI. A web interface built with React is planned for broader usability.  

Setup is straightforward: clone the repository, create a virtual environment, install dependencies from `requirements.txt`, and add a `.env` file with your OpenAI API key. The backend can be launched with Uvicorn on port 8000, and the frontend can be extended to connect seamlessly to the API.  

Future improvements include scaling the model with a larger dataset, refining multilingual support, and exploring direct integration with SAP systems. This project is released under the MIT License.  

## 📄 License Notice
This project builds upon [NanoGPT](https://github.com/karpathy/nanoGPT),  
which is distributed under the [MIT License](https://opensource.org/licenses/MIT).

