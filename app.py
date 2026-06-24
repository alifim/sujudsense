import asyncio

import chainlit as cl
from engine import SujudSenseEngine
from logger import logger

ENGINE_INSTANCE: SujudSenseEngine | None = None
ENGINE_INIT_LOCK = asyncio.Lock()
ENGINE_WARM_SCHEDULED = False


@cl.set_starters
async def set_starters(user: cl.User | None = None, conversation: str | None = None):
    return [
        cl.Starter(
            label="Knee pain guidance",
            message="My knee is hurt. How should I perform prayer?",
            icon="/public/knee.svg",
        ),

        cl.Starter(
            label="Lower back adjustments",
            message="I feel lower back pain during Ruku; what adjustments are safe?",
            icon="/public/back.svg",
        ),

        cl.Starter(
            label="Sujud reach issues",
            message="My palms don't reach the ground in Sujud — what can I do?",
            icon="/public/sujud.svg",
        ),

        cl.Starter(
            label="Chair guidance",
            message="When should I sit on a chair instead of performing Sujud?",
            icon="/public/sit.svg",
        ),
    ]

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
    # Do not send UI messages here to avoid bypassing starters.
    # If the engine is already warmed, bind it to the user session.
    engine = ENGINE_INSTANCE
    if engine is not None:
        cl.user_session.set("engine", engine)
        return

    # Otherwise, schedule a background warm-up once and avoid awaiting here.
    global ENGINE_WARM_SCHEDULED
    if not ENGINE_WARM_SCHEDULED:
        asyncio.create_task(get_engine())
        ENGINE_WARM_SCHEDULED = True

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