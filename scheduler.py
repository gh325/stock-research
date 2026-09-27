"""
scheduler.py — Nightly automated pipeline using APScheduler.

Runs the full pipeline every evening at SCHEDULE_HOUR:SCHEDULE_MINUTE
(default: 9:00 PM local time), after US markets close at 4 PM ET.
"""
import logging

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from config import SCHEDULE_HOUR, SCHEDULE_MINUTE

logger = logging.getLogger(__name__)


def start_scheduler(pipeline_fn) -> None:
    """
    Start the blocking scheduler. The process stays alive until Ctrl-C.
    pipeline_fn: callable — e.g. run_pipeline from main.py
    """
    scheduler = BlockingScheduler(timezone="America/New_York")
    scheduler.add_job(
        func=pipeline_fn,
        trigger=CronTrigger(hour=SCHEDULE_HOUR, minute=SCHEDULE_MINUTE),
        id="nightly_research",
        name="Nightly stock research pipeline",
        misfire_grace_time=3_600,   # run up to 1 hr late if machine was asleep
        replace_existing=True,
    )

    next_run = scheduler.get_jobs()[0].next_run_time
    logger.info(f"Scheduler started. Next run: {next_run}")
    print(f"\nScheduler running — next analysis at {SCHEDULE_HOUR:02d}:{SCHEDULE_MINUTE:02d} ET.")
    print("Press Ctrl-C to stop.\n")

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
        scheduler.shutdown(wait=False)
