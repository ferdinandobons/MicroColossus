# MicroColossus

> **Trade memory for time.**

Runtime sperimentale open source per addestrare modelli che superano la RAM e la VRAM disponibili, usando streaming gerarchico dei tensori, ricomputazione, offloading su NVMe e tiling interno agli strati.

> Stato: specifica iniziale. L'implementazione non esiste ancora e non vengono dichiarati risultati prestazionali.

## Visione

MicroColossus tratta VRAM, RAM e NVMe come una gerarchia di memoria esplicita:

1. **VRAM** come cache L1 e spazio di calcolo.
2. **RAM** come cache L2 limitata, staging area e memoria pinned controllata.
3. **NVMe** come archivio canonico di pesi, gradienti, stati dell'ottimizzatore e checkpoint.

Il runtime mantiene sulla GPU soltanto il working set necessario, sovrappone trasferimenti e calcolo, ricalcola alcune attivazioni e divide anche un singolo strato in tile quando non entra in VRAM.

L'obiettivo non è creare modelli infiniti. È spostare il limite immediato della memoria verso capacità dello storage, banda, compute, tempo disponibile ed endurance dell'SSD.

## Problema

Durante il training non si conservano soltanto i pesi. Servono anche gradienti, stati dell'ottimizzatore, eventuali copie master, attivazioni e workspace. Una configurazione AdamW mixed precision può richiedere indicativamente circa 16 byte o più per parametro, prima delle attivazioni.

| Parametri | Stato persistente indicativo |
|---:|---:|
| 124 milioni | circa 2,0 GB |
| 350 milioni | circa 5,6 GB |
| 1 miliardo | circa 16 GB |
| 7 miliardi | circa 112 GB |

La domanda centrale è:

> Come eseguire un aggiornamento matematicamente valido quando lo stato del modello non entra né in VRAM né nella RAM disponibile al processo?

## Ipotesi centrale

Per una classe utile di modelli e ricette di training, una parte rilevante del requisito di memoria residente può essere sostituita con trasferimenti, ricomputazione e tempo, mantenendo aggiornamenti full-parameter e un comportamento numerico confrontabile con una baseline residente.

Vincoli fondamentali:

```text
stato modello + checkpoint + metadati <= capacità NVMe allocata

tile parametri + attivazioni + workspace + buffer <= budget VRAM
```

## Obiettivi

- Training di modelli il cui stato completo supera VRAM e budget RAM.
- Modalità full-parameter di riferimento.
- Budget rigidi per VRAM, RAM, NVMe e scritture SSD.
- Target iniziale: una GPU CUDA da 8 GB, 8 GB di RAM installata e un singolo NVMe consumer.
- Tiling intra-layer per matrici, embedding, logits, normalizzazioni e attenzione.
- I/O asincrono con overlap fra GPU, PCIe, CPU e NVMe.
- Journal, versionamento dei tensori, checkpoint atomici e crash recovery.
- Telemetria completa per memoria, I/O, utilizzo GPU e write amplification.

## Non obiettivi

- Modelli letteralmente infiniti.
- Throughput equivalente a cluster multi-GPU.
- Supporto iniziale a ogni operatore PyTorch.
- Uso dello swap o della page cache senza contabilità.
- Presentare offloading, checkpointing o quantizzazione come idee nuove.

## Modalità previste

### `reference`

Tutti i parametri sono addestrabili. Usa pesi e gradienti completi, AdamW o SGD, activation checkpointing e tiling matematicamente equivalente. Non introduce intenzionalmente adapter o proiezioni low-rank.

### `compact`

Può usare optimizer states quantizzati, gradienti compressi, GaLore o altre approssimazioni. Ogni deviazione dalla modalità di riferimento deve essere dichiarata e misurata.

### `adapter`

Percorso futuro per LoRA e QLoRA, distinto dal full training.

## Architettura proposta

```text
Dataset stream
      |
Graph capture / frontend
      |
Budget-aware planner
      |
Execution schedule
      |
+-------------------------------+
| Runtime                       |
| - executor                    |
| - transfer engine             |
| - activation manager          |
| - optimizer engine            |
| - checkpoint coordinator      |
+-------------------------------+
      |
VRAM cache <-> RAM cache <-> NVMe tensor store
```

Componenti principali:

- **Frontend** per un decoder Transformer controllato e successivi adapter PyTorch.
- **Tensor manifest** con forma, dtype, versione, posizione, checksum e prossima posizione d'uso.
- **Planner** che seleziona tile, checkpoint, offload e prefetch rispettando hard budget.
- **Transfer engine** asincrono con buffer preallocati e pinned memory limitata.
- **Tensor store** chunked e transazionale su NVMe.
- **Executor** per forward, backward e optimizer step layer-wise o tile-wise.
- **Telemetry engine** per picchi, byte trasferiti, tempi, stall ed endurance.

## Modello di esecuzione

### Forward

1. Prefetch dei pesi del prossimo layer o tile da NVMe a RAM.
2. Trasferimento RAM-VRAM.
3. Calcolo GPU.
4. Conservazione, offload o eliminazione dell'attivazione secondo il piano.
5. Rilascio del working set.

### Backward

1. Ricaricamento dei pesi necessari.
2. Recupero o ricomputazione delle attivazioni.
3. Calcolo dei gradienti.
4. Scaricamento e accumulo per chunk.
5. Aggiornamento dell'ottimizzatore su CPU o GPU per tile.
6. Pubblicazione atomica della nuova versione.

## Tiling intra-layer

Il solo layer streaming non basta quando un singolo layer è più grande della VRAM. MicroColossus dovrà supportare:

- Linear e MLP tiled.
- Embedding a partizioni.
- Output head e cross-entropy senza materializzare tutti i logits.
- Attention a blocchi.
- LayerNorm e riduzioni streaming.
- Weight tying fra embedding e output head.
- Dropout riproducibile durante la ricomputazione.

## Planner e modello di costo

Configurazione indicativa:

```yaml
hardware:
  vram_budget_gib: 7.0
  ram_budget_gib: 5.0
  nvme_budget_gib: 500
  ssd_write_budget_tb: 300

training:
  mode: reference
  micro_batch_size: 1
  sequence_length: 1024
  gradient_accumulation_steps: 8

planner:
  objective: min_step_time
  enforce_hard_budgets: true
  account_page_cache: true
```

Limite inferiore ideale del tempo per step:

```text
time_step >= max(
  FLOP_step / throughput_GPU,
  byte_PCIe / banda_PCIe,
  byte_NVMe / banda_NVMe,
  lavoro_optimizer / throughput_CPU
)
```

Il planner deve anche minimizzare le scritture e stimare l'endurance consumata.

## Consistenza e recovery

- Chunk immutabili o copy-on-write.
- Manifest con versioni logiche.
- Journal write-ahead.
- Checksum per chunk.
- Commit atomico dello step.
- Checkpoint incrementali.
- Failure injection nei test.

Uno step è valido soltanto quando tutti i tensori richiesti e il manifest della nuova versione sono stati pubblicati correttamente.

## MVP

### Fase 0. Simulator

Planner senza training reale, con modelli di capacità, latenza, banda e scritture.

### Fase 1. Baseline numerica

Decoder circa 124M residente e streamed. Confronto di loss, gradienti e parametri.

### Fase 2. Tensor store

Chunking, manifest, checksum, journal, recovery e telemetria I/O.

### Fase 3. Streaming layer-wise

Pesi e optimizer states su NVMe, cache RAM limitata e doppio buffering.

### Fase 4. Tiling intra-layer

Linear, MLP, embedding, output head e attention tiled.

### Fase 5. Dimostrazione

Full-parameter training circa 350M su GPU da 8 GB, RAM fortemente limitata e singolo NVMe. Stretch target circa 1B.

## Benchmark

Confronti previsti con:

- PyTorch residente.
- PyTorch activation checkpointing.
- PyTorch CPU offload quando applicabile.
- DeepSpeed ZeRO-Offload e ZeRO-Infinity.
- MicroColossus `reference`.
- MicroColossus `compact`, separatamente.

Metriche principali:

- picco VRAM e RSS;
- token al secondo e secondi per step;
- utilizzo GPU;
- tempo in attesa I/O;
- byte PCIe e NVMe;
- write amplification;
- tempo di checkpoint e recovery;
- differenza numerica dalla baseline;
- energia per milione di token, quando misurabile.

## Stack iniziale

- Python 3.11+
- PyTorch 2.x
- C++ e CUDA
- io_uring o libaio
- YAML e JSONL
- pytest, Ruff, mypy e pre-commit
- GitHub Actions

## Struttura prevista

```text
MicroColossus/
  README.md
  LICENSE
  pyproject.toml
  microcolossus/
    frontend/
    graph/
    planner/
    runtime/
    storage/
    kernels/
    telemetry/
  tests/
  benchmarks/
  docs/
  examples/
  tools/
```

## Lavori correlati

- [PyTorch activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html)
- [ZeRO-Offload](https://arxiv.org/abs/2101.06840)
- [ZeRO-Infinity](https://arxiv.org/abs/2104.07857)
- [QLoRA](https://arxiv.org/abs/2305.14314)
- [GaLore](https://arxiv.org/abs/2403.03507)
- [LoHan](https://arxiv.org/abs/2403.06504)

## Differenziazione proposta

- Target esplicito da 8 GB di RAM installata, non soltanto VRAM ridotta.
- Hard budget applicati a ogni tier.
- NVMe come tensor store canonico e transazionale.
- Tiling anche dentro il singolo strato.
- Planner che considera tempo e scritture SSD.
- Telemetria per byte per token e write amplification.
- Crash recovery integrato nel modello di esecuzione.
- Separazione rigorosa fra `reference`, `compact` e `adapter`.

Questa differenziazione deve essere validata sperimentalmente. Non è una dichiarazione di novità scientifica già dimostrata.

## Licenza

Apache License 2.0. Consultare [`LICENSE`](LICENSE).

## Nota finale

```text
meno memoria residente
        in cambio di
più I/O + più ricomputazione + più tempo
```

Il primo risultato importante non sarà il modello più grande possibile. Sarà un runtime capace di spiegare dove vive ogni tensore, quanto costa spostarlo, quanto spazio richiede e quale versione è valida.
