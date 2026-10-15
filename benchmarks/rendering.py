"""Common offscreen renderer; contains no ABM transition logic."""
from PIL import Image, ImageDraw
import math


def draw_state(state, engine):
    if 'forager.collected' not in state:
        return draw_agents(state, engine)
    values = state['forager.collected']
    columns = min(16, len(values))
    rows = math.ceil(len(values)/columns)
    width, cell = 800, 48
    canvas = Image.new('RGB', (width, 100+rows*cell), '#101820')
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 15), 'Shared-resource ABM | ' + engine.upper(), fill='white')
    draw.text((20, 38), f"Step: {state['global.rounds']}   Free resources: {state['global.available']}", fill='white')
    draw.text((20, 61), 'Each cell is one agent. Number = cumulative collection.', fill='#a0b0bf')
    for index, count in enumerate(values):
        x, y = 16+(index % columns)*cell, 90+(index//columns)*cell
        draw.rectangle((x, y, x+cell-3, y+cell-3), fill='#32a77e' if count else '#354454')
        draw.text((x+4, y+3), str(index), fill='#bcd4d4')
        draw.text((x+4, y+23), str(count), fill='white')
    return canvas


def draw_agents(state, engine):
    """Generic observed-state display: coordinates, ring positions or stored graph edges."""
    canvas = Image.new('RGB', (800, 800), '#101820')
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 15), 'Observed ABM | ' + engine.upper(), fill='white')
    draw.text((20, 38), f"Step: {state.get('global.ticks', 0)} | Common offscreen renderer", fill='#a0b0bf')
    species = sorted({key.split('.')[0] for key in state if not key.startswith('global.')}, key=lambda name: (name != 'patch', name))
    # Fixed source-defined benchmark domains; no normalization that could hide scale drift.
    domain = 100 if any(k.endswith('.location') for k in state) else 20 if any(k.endswith(('.vx', '.evacuated')) for k in state) else 8
    palette = ['#32a77e', '#fa9f42', '#6db1ff', '#df7bc8']
    def position(name, index, count):
        if name+'.location' in state:
            point = state[name+'.location'][index]
            return 40 + point[0] / domain * 700, 90 + point[1] / domain * 660
        if name+'.px' in state:
            return 40 + state[name+'.px'][index] / domain * 700, 90 + state[name+'.py'][index] / domain * 660
        if name+'.position' in state:
            angle = state[name+'.position'][index] / 128 * 2 * math.pi
        else: angle = index / count * 2 * math.pi
        return 400 + 280 * math.cos(angle), 420 + 280 * math.sin(angle)
    for order, name in enumerate(species):
        values = next(v for k, v in state.items() if k.startswith(name+'.'))
        count = len(values)
        if name+'.link_a' in state:
            for index in range(count):
                for field in ('link_a', 'link_b'):
                    target = state[name+'.'+field][index]
                    draw.line([position(name,index,count), position(name,target,count)], fill='#354454', width=1)
        for index in range(count):
            if name+'.active' in state and not state[name+'.active'][index]: continue
            x, y = position(name,index,count)
            color = palette[order % len(palette)]
            if name+'.status' in state:
                color = ['#354454','#32a77e','#f05b57'][state[name+'.status'][index]]
            if name+'.group_id' in state: color = palette[state[name+'.group_id'][index]]
            if name+'.adopted' in state: color = '#fa9f42' if state[name+'.adopted'][index] else '#354454'
            if name+'.carrying' in state: color = '#fa9f42' if state[name+'.carrying'][index] else color
            if name+'.trail' in state:
                amount = min(1.0,state[name+'.trail'][index])
                color = (int(35+120*amount),int(55+150*amount),70)
            size = 7 if name != 'patch' else 12
            draw.ellipse((x-size,y-size,x+size,y+size),fill=color)
        draw.text((20, 730 + order*15), f'{name}: {count} fixed slots', fill=palette[order % len(palette)])
    return canvas
