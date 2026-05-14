# Sheet Resistance Device Integration

This folder now contains a PyQt5 app and support modules for integrating the SURAGUS `EddyCus TF lab 2020` over Ethernet.

## What is confirmed

- The connected PC interface is on link-local IPv4: `169.254.181.167/16`.
- The Ethernet link is physically up, but Windows has not learned a peer IP or MAC address yet.
- SURAGUS publicly states that the `EddyCus lab 2020` series can be operated either by vendor software or by customer software via SDK.
- I did not find a public download for the vendor software or a public API reference for the exact `TF lab 2020` unit.
- SURAGUS public integration material for related systems mentions a REST API for measurement electronics. That is a useful lead, but it is not proof that this exact unit uses the same interface.

## Practical next steps

1. Ask SURAGUS support for the exact Windows control software for your serial number, or for the SDK/API documentation for the `EddyCus TF lab 2020`.
2. If the device has a service screen or front-panel network menu, check whether it is using:
   - DHCP
   - link-local (`169.254.x.x`)
   - a fixed static IP
3. Use the discovery script here to scan the likely subnet and probe common HTTP/REST ports.

## Files

- `sheet_resistance_app.py`: the main PyQt5 UI, following the same grouped-window style used in the Keithley tools.
- `transport.py`: discovery and transport backends for `Mock` and configurable `HTTP/REST` operation.
- `discover.py`: a CLI scanner for finding candidate hosts.
- `suragus_client.py`: low-level HTTP/REST probing helpers.

## Usage

Launch the GUI from the repository root:

```powershell
python SheetResistance\sheet_resistance_app.py
```

The GUI supports:

- `Mock` mode for immediate testing.
- `HTTP/REST` mode for device discovery, connection, repeated measurement, live plotting, and CSV logging.
- configurable status path, measurement path, method, JSON payload, and response key mapping.

Run a focused scan first if you want to identify the host before using the GUI:

```powershell
python SheetResistance\discover.py --cidr 169.254.181.0/24
```

If nothing appears, widen the search:

```powershell
python SheetResistance\discover.py --cidr 169.254.0.0/16 --limit 4096
```

Probe a specific host if you find one:

```powershell
python SheetResistance\discover.py --host 169.254.181.50
```

## Notes

- The GUI can issue repeated measurement requests in `HTTP/REST` mode, but the exact `TF lab 2020` endpoint and response schema are still not publicly documented.
- `Mock` mode is fully usable now.
- For the real unit, you will likely need to adjust `Status path`, `Measure path`, `Method`, `JSON payload`, `Value key`, and `Unit key`.
- If the unit uses a non-HTTP protocol, this app still gives you a clean UI shell to swap in a different transport backend once the actual interface is known.
