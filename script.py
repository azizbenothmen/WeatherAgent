from __future__ import annotations as _annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import logfire
from httpx import AsyncClient
from pydantic import BaseModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.models.openai import OpenAIChatModel


from pydantic_ai import Agent, RunContext
import os
from dotenv import load_dotenv

load_dotenv()

logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()


@dataclass
class Deps:
    client: AsyncClient

provider = OpenRouterProvider(
    api_key=os.getenv('OPEN_ROUTER_KEY'),
)
model = OpenAIChatModel('apodex/apodex-1.1-mini:free', provider=provider)

weather_agent = Agent(
    model,
    
    instructions='Be concise, reply with one sentence.',
    deps_type=Deps,
    retries=2,
)


class LatLng(BaseModel):
    lat: float
    lng: float


@weather_agent.tool
async def get_lat_lng(ctx: RunContext[Deps], location_description: str) -> LatLng:
    """Get the latitude and longitude of a location.

    Args:
        ctx: The context.
        location_description: A description of a location.
    """
    r = await ctx.deps.client.get(
        'https://api.weatherapi.com/v1/forecast.json',
        params={'key':os.getenv('WEATHER_KEY'),'q': location_description,'days': 2, 'aqi': 'no'},
    )
    r.raise_for_status()
    return LatLng.model_validate_json(r.content)


@weather_agent.tool
async def get_weather(ctx: RunContext[Deps], lat: float, lng: float,location_description: str) -> dict[str, Any]:
    """Get the weather at a location.

    Args:
        ctx: The context.
        lat: Latitude of the location.
        lng: Longitude of the location.
    """
    temp_response, descr_response = await asyncio.gather(
        ctx.deps.client.get(
            'https://api.weatherapi.com/v1/forecast.json',
                    params={'key':os.getenv('WEATHER_KEY'),'q': location_description,'days':2, 'aqi': 'no'},
        ),
        ctx.deps.client.get(
            'https://api.weatherapi.com/v1/forecast.json',
                    params={'key':os.getenv('WEATHER_KEY'),'q': location_description,'days': 2, 'aqi': 'no'},
        ),
    )
    temp_response.raise_for_status()
    descr_response.raise_for_status()
    return {
        'temperature': f'{temp_response.text} °C',
        'description': descr_response.text,
    }


async def main():
    async with AsyncClient() as client:
        logfire.instrument_httpx(client, capture_all=True)
        deps = Deps(client=client)
        result = await weather_agent.run(
            'What is the weather like in Tunisia Tomorrow and in Canada?', deps=deps
        )
        print('Response:', result.output)


if __name__ == '__main__':
    asyncio.run(main())