import asyncio

import chainlit as cl
from engine import SujudSenseEngine
from logger import logger

ENGINE_INSTANCE: SujudSenseEngine | None = None
ENGINE_INIT_LOCK = asyncio.Lock()

async def get_engine() -> SujudSenseEngine:
    global ENGINE_INSTANCE
    if ENGINE_INSTANCE is not None:
        return ENGINE_INSTANCE

    async with ENGINE_INIT_LOCK:
        if ENGINE_INSTANCE is not None:
            return ENGINE_INSTANCE

        engine = SujudSenseEngine()
        await engine.initialize()
        ENGINE_INSTANCE = engine
        return ENGINE_INSTANCE

@cl.on_chat_start
async def setup_chat():
    """Initializes the UI and binds the engine instance to the session."""
    msg = cl.Message(content="Welcome to **SujudSense**. Initializing system...")
    await msg.send()

    try:
        engine = await get_engine()
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
        engine = await get_engine()
        cl.user_session.set("engine", engine)

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