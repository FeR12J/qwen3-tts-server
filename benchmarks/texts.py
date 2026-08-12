"""Textos de referencia para los benchmarks.

12 textos: 4 cortos, 4 medios y 4 largos, repartidos entre español e inglés,
más una bateria de textos lingüísticamente difíciles.

Cada texto se identifica con una clave única (id) que se usa en los
resultados JSON y en los informes.
"""

# Duraciones aproximadas (habla normal) por categoría:
#   short  ~  5 -  10 s
#   medium ~ 15 -  30 s
#   long   ~ 40 -  70 s

SHORT_TEXTS = [
    {
        "id": "es_short_01",
        "lang": "es",
        "category": "short",
        "text": "La inteligencia artificial está transformando la forma en que utilizamos el software.",
    },
    {
        "id": "es_short_02",
        "lang": "es",
        "category": "short",
        "text": "Los modelos locales permiten ejecutar inferencia sin depender de servicios externos.",
    },
    {
        "id": "en_short_01",
        "lang": "en",
        "category": "short",
        "text": "Local language models can run inference directly on consumer hardware.",
    },
    {
        "id": "en_short_02",
        "lang": "en",
        "category": "short",
        "text": "Memory bandwidth is often a critical factor in local AI inference.",
    },
]

MEDIUM_TEXTS = [
    {
        "id": "es_medium_01",
        "lang": "es",
        "category": "medium",
        "text": (
            "Ejecutar un modelo de lenguaje local requiere equilibrar memoria, ancho de banda "
            "y capacidad de cómputo. Una GPU con suficiente VRAM puede ejecutar modelos mucho "
            "más grandes, pero la velocidad de generación también depende de cómo se almacenan "
            "y procesan los pesos y la caché KV."
        ),
    },
    {
        "id": "es_medium_02",
        "lang": "es",
        "category": "medium",
        "text": (
            "Un sistema de agentes puede combinar un modelo de lenguaje con herramientas "
            "externas, memoria y mecanismos de planificación. La dificultad no está únicamente "
            "en generar texto, sino en coordinar correctamente las distintas operaciones y "
            "gestionar los errores que aparecen durante la ejecución."
        ),
    },
    {
        "id": "en_medium_01",
        "lang": "en",
        "category": "medium",
        "text": (
            "Local inference involves several competing constraints: model size, available "
            "memory, memory bandwidth, compute throughput, and context length. Increasing the "
            "context window can significantly increase the memory required by the KV cache, "
            "especially when serving multiple requests concurrently."
        ),
    },
    {
        "id": "en_medium_02",
        "lang": "en",
        "category": "medium",
        "text": (
            "An agent can combine a language model with external tools, memory, and a planning "
            "mechanism. The model itself does not execute every operation directly; instead, "
            "it decides which action to take and uses the result to continue the task."
        ),
    },
]

LONG_TEXTS = [
    {
        "id": "es_long_01",
        "lang": "es",
        "category": "long",
        "text": (
            "La inferencia local de modelos de lenguaje presenta un problema de ingeniería "
            "diferente al entrenamiento. Durante la generación, los pesos del modelo deben "
            "permanecer disponibles y cada nuevo token requiere acceder repetidamente a una "
            "parte importante de esos datos. Por este motivo, el ancho de banda de memoria "
            "puede convertirse en una limitación tan importante como la capacidad de cómputo.\n\n"
            "La cantidad de memoria disponible también determina qué modelos pueden ejecutarse. "
            "La cuantización permite reducir el tamaño de los pesos, aunque la reducción de "
            "precisión puede afectar de forma diferente a cada tarea. Además, durante una "
            "conversación larga, la caché KV puede representar una parte considerable de la "
            "memoria utilizada por el sistema.\n\n"
            "Por tanto, elegir un modelo local no consiste simplemente en buscar el modelo con "
            "más parámetros. Hay que considerar simultáneamente memoria, ancho de banda, "
            "precisión, longitud de contexto y velocidad de generación."
        ),
    },
    {
        "id": "es_long_02",
        "lang": "es",
        "category": "long",
        "text": (
            "Un sistema agéntico sencillo puede construirse alrededor de un ciclo relativamente "
            "pequeño. El modelo recibe el estado actual, determina qué acción necesita realizar "
            "y, cuando dispone de herramientas, solicita su ejecución. El resultado de la "
            "herramienta vuelve al contexto y el modelo puede utilizarlo para decidir el "
            "siguiente paso.\n\n"
            "Esta arquitectura parece sencilla, pero introduce varios problemas prácticos. Las "
            "herramientas pueden fallar, devolver resultados inesperados o tardar más de lo "
            "previsto. El contexto puede crecer rápidamente y las acciones pueden necesitar "
            "validación antes de ejecutarse. Además, no todas las tareas requieren el mismo "
            "número de pasos.\n\n"
            "Por eso resulta útil separar las responsabilidades del sistema. El modelo se ocupa "
            "de la decisión lingüística, mientras que el runtime controla las herramientas, los "
            "estados y los límites de ejecución. Esta separación permite medir cada componente "
            "de forma independiente y facilita la depuración cuando el agente produce un "
            "resultado incorrecto."
        ),
    },
    {
        "id": "en_long_01",
        "lang": "en",
        "category": "long",
        "text": (
            "Local inference of language models presents a different engineering problem than "
            "training. During generation, the model weights must remain available and each new "
            "token requires repeatedly accessing a significant portion of that data. For this "
            "reason, memory bandwidth can become as important a bottleneck as raw compute.\n\n"
            "The amount of available memory also determines which models can run. Quantization "
            "reduces the size of the weights, although the loss of precision can affect "
            "different tasks in different ways. Moreover, during a long conversation, the KV "
            "cache can represent a large fraction of the memory used by the system.\n\n"
            "Thus, choosing a local model is not simply about picking the one with the most "
            "parameters. You must consider memory, bandwidth, precision, context length, and "
            "generation speed all at the same time."
        ),
    },
    {
        "id": "en_long_02",
        "lang": "en",
        "category": "long",
        "text": (
            "A simple agentic system can be built around a relatively small loop. The model "
            "receives the current state, decides which action it needs to take, and, when tools "
            "are available, requests their execution. The tool result returns to the context "
            "and the model can use it to decide the next step.\n\n"
            "This architecture looks simple, but it introduces several practical issues. Tools "
            "can fail, return unexpected results, or take longer than expected. The context "
            "can grow quickly, and actions may need validation before being executed. Also, "
            "not every task requires the same number of steps.\n\n"
            "That is why it helps to separate responsibilities. The model handles the "
            "linguistic decision, while the runtime controls tools, states, and execution "
            "limits. This separation lets you measure each component independently and makes "
            "debugging easier when the agent produces a wrong result."
        ),
    },
]

# Textos lingüísticamente difíciles: números, fechas, monedas, cifras,
# signos, preguntas, abreviaturas y mezcla de español e inglés.
HARD_TEXTS = [
    {
        "id": "es_hard_01",
        "lang": "es",
        "category": "hard",
        "text": "El pedido número 4827 cuesta 129,95 euros y llegará entre el 14 y el 18 de septiembre de 2026.",
    },
    {
        "id": "es_hard_02",
        "lang": "es",
        "category": "hard",
        "text": "El servidor utiliza una API REST con FastAPI y puede integrarse con OpenWebUI mediante un endpoint compatible con TTS.",
    },
    {
        "id": "es_hard_03",
        "lang": "es",
        "category": "hard",
        "text": "¿Realmente necesitamos una GPU? Depende. Para un modelo pequeño, quizá no; para uno de 70B, la respuesta cambia completamente.",
    },
    {
        "id": "es_hard_04",
        "lang": "es",
        "category": "hard",
        "text": "La GPU dispone de 24 GB de VRAM y alcanza aproximadamente 1,2 TB/s de ancho de banda de memoria.",
    },
    {
        "id": "en_hard_01",
        "lang": "en",
        "category": "hard",
        "text": "The 4827th order costs 129.95 euros and will arrive between September 14 and 18, 2026.",
    },
    {
        "id": "en_hard_02",
        "lang": "en",
        "category": "hard",
        "text": "The server exposes a REST API with FastAPI and integrates with OpenWebUI through a TTS-compatible endpoint.",
    },
    {
        "id": "en_hard_03",
        "lang": "en",
        "category": "hard",
        "text": "Do we really need a GPU? It depends. For a small model, probably not; for a 70B one, the answer changes completely.",
    },
    {
        "id": "en_hard_04",
        "lang": "en",
        "category": "hard",
        "text": "The GPU has 24 GB of VRAM and sustains roughly 1.2 TB/s of memory bandwidth.",
    },
    {
        "id": "es_hard_05",
        "lang": "es",
        "category": "hard",
        "text": "La GPU dispone de 24 GB de VRAM y alcanza aproximadamente 1,2 TB/s de ancho de banda de memoria.",
    },
    {
        "id": "es_hard_06",
        "lang": "es",
        "category": "hard",
        "text": "Bueno, la respuesta corta es que sí funciona, pero hay una diferencia bastante grande entre ejecutarlo en CPU y utilizar una GPU.",
    },
]

ALL_TEXTS = SHORT_TEXTS + MEDIUM_TEXTS + LONG_TEXTS + HARD_TEXTS


def all_texts():
    return list(ALL_TEXTS)


def by_category(category):
    return [t for t in ALL_TEXTS if t["category"] == category]
