import chainlit as cl
from engine import SujudSenseEngine
from logger import logger

@cl.on_chat_start
async def setup_chat():
    """Initializes the UI and binds the engine instance to the session."""
    msg = cl.Message(content="Welcome to **SujudSense**. Initializing system...")
    await msg.send()

    try:
        # Instantiate and initialize the business logic service
        engine = SujudSenseEngine()
        await engine.initialize()
        
        # Store the engine instance in the session
        cl.user_session.set("engine", engine)
        
        msg.content = "Welcome to **SujudSense**. How can I help you safely adjust your prayer posture today?"
        await msg.update()
    except Exception as e:
        logger.error(f"Failed to bind engine to session: {str(e)}", exc_info=True)
        msg.content = "System failure. Please contact support."
        await msg.update()

@cl.on_message
async def handle_query(message: cl.Message):
    """Passes user input to the engine and returns the response asynchronously."""
    engine = cl.user_session.get("engine")
    if engine is None:
        await cl.Message(content="Session expired. Please refresh the page.").send()
        return

    msg = cl.Message(content="Analyzing...")
    await msg.send()

    try:
        # 1. Non-blocking Firewall Check
        is_safe = await engine.check_firewall(message.content)
        if not is_safe:
            logger.info(f"Firewall blocked query: '{message.content}'")
            msg.content = "I am SujudSense, specialized exclusively in prayer biomechanics and Fiqh. I cannot assist with off-topic conversations."
            await msg.update()
            return

        # 2. Non-blocking LLM Generation
        answer = await engine.generate_response(message.content)
        msg.content = answer
        await msg.update()

    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}", exc_info=True)
        msg.content = "An error occurred while analyzing your request."
        await msg.update()