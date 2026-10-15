"""Native GAMA adapter with no JAX import (also used by isolated benchmarks)."""
from pathlib import Path
import json
import hashlib
import uuid
import websocket


class GamaClient:
    def __init__(self, url='ws://127.0.0.1:6868', timeout=120, *, record_response_payloads=True):
        self.socket = websocket.create_connection(url, timeout=timeout)
        message = json.loads(self.socket.recv())
        if message.get('type') != 'ConnectionSuccessful':
            self.socket.close()
            raise RuntimeError(f'Unexpected GAMA greeting: {message}')
        self.greeting = message
        self.commands = []
        self.record_response_payloads=record_response_payloads

    def close(self): self.socket.close()

    def command(self, type_, **parameters):
        request = dict(type=type_, request_id=str(uuid.uuid4()), **parameters)
        self.socket.send(json.dumps(request))
        while True:
            raw=self.socket.recv();message = json.loads(raw)
            if message.get('command', {}).get('request_id') == request['request_id']:
                response=message
                if not self.record_response_payloads and message.get('type')=='CommandExecutedSuccessfully':
                    encoded=raw.encode('utf-8') if isinstance(raw,str) else raw
                    response={'type':message['type'],'payload_bytes':len(encoded),'payload_sha256':hashlib.sha256(encoded).hexdigest()}
                self.commands.append(dict(request=request, response=response))
                if message.get('type') != 'CommandExecutedSuccessfully':
                    raise RuntimeError(f'GAMA rejected command: {message}')
                return message.get('content')
            if message.get('type') in ('SimulationError', 'GamaServerError'):
                raise RuntimeError(f'GAMA runtime failure: {message}')

    def expression(self, exp_id, expression):
        value = self.command('expression', exp_id=exp_id, expr=expression)
        # GAMA releases may return JSON values or their serialized GAML representation.
        if isinstance(value, str):
            try: return json.loads(value)
            except json.JSONDecodeError as error:
                raise RuntimeError(f'Non-JSON numeric GAMA result: {value!r}') from error
        return value

    def snapshot(self, exp_id, manifest):
        """Observe all attributes in one RPC, preserving float round-trip precision."""
        entries = []
        for key, type_ in manifest['fields'].items():
            scope, field = key.split('.')
            if scope == 'global':
                expr = f'string(simulation.{field})' if type_ == 'float' else f'simulation.{field}'
            else:
                value = f'string(each.{field})' if type_ == 'float' else f'each.{field}'
                expr = f'({scope} sort_by (each.index)) collect ({value})'
            if type_ == 'point':
                point = f'simulation.{field}' if scope == 'global' else f'each.{field}'
                coordinates = '[' + ', '.join(f'string({point}.{axis})' for axis in ('x', 'y', 'z')) + ']'
                expr = coordinates if scope == 'global' else f'({scope} sort_by (each.index)) collect ({coordinates})'
            if type_.startswith('agent:'):
                reference = f'simulation.{field}' if scope=='global' else f'each.{field}'
                value = f'({reference}=nil ? -1 : {reference}.index)'
                expr = value if scope=='global' else f'({scope} sort_by (each.index)) collect ({value})'
            if type_.startswith('agents:'):
                collection = f'simulation.{field}' if scope=='global' else f'each.{field}'
                value = f'({collection} collect each.index)'
                expr = value if scope=='global' else f'({scope} sort_by (each.index)) collect ({value})'
            if type_ == 'rgb':
                color = f'simulation.{field}' if scope=='global' else f'each.{field}'
                channels = '['+', '.join(f'{color}.{channel}' for channel in ('red','green','blue','alpha'))+']'
                expr = channels if scope=='global' else f'({scope} sort_by (each.index)) collect ({channels})'
            if type_ == 'geometry':
                expr = f'simulation.{field}.points collect [string(each.x),string(each.y),string(each.z)]'
            entries.append(f"'{key}' :: ({expr})")
        result = self.expression(exp_id, '[' + ', '.join(entries) + ']')
        if not isinstance(result, dict) or set(result) != set(manifest['fields']):
            raise RuntimeError(f'Unexpected GAMA snapshot schema: {result!r}')
        for key, type_ in manifest['fields'].items():
            if type_ in ('float', 'point', 'geometry'):
                def convert(value):
                    return [convert(v) for v in value] if isinstance(value, list) else float(value)
                result[key] = convert(result[key])
        return result

    def trace(self, source, manifest, steps, experiment):
        if experiment not in manifest['experiments']:
            raise ValueError(f'Unknown experiment {experiment!r}')
        exp_id = self.command('load', model=str(Path(source).resolve()), experiment=experiment,
                              console=False, runtime=True)
        try:
            result = []
            for step in range(steps+1):
                if step:
                    self.command('step', exp_id=exp_id, nb_step=1, sync=True)
                state = {}
                for species, count in manifest['populations'].items():
                    indices = self.expression(exp_id, f'({species} sort_by (each.index)) collect (each.index)')
                    if indices != list(range(count)):
                        raise RuntimeError(f'GAMA agent identity differs for {species}: {indices}')
                state = self.snapshot(exp_id, manifest)
                result.append(state)
            return result
        finally:
            self.command('stop', exp_id=exp_id)

