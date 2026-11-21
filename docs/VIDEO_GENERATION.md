# ACOT Video Generation

Status: design agreed, implementation not started. The module folder exists with empty files.
Last updated: 21 Nov 2026.

This document covers the feature from the first idea to where it stands now: what it is, how the decisions changed, what the data looks like (measured on `acotdb`), the stack and cost, the backend design, risks, and next steps.

---

## 1. What the feature is

An agent, inside the agent-lead chat, clicks **Generate video**. ACOT builds a **60-second property explainer video with a spoken explanation**, based on the properties and communities the buyer searched for in ACOT AI chat. The finished video goes back to the agent, who previews it, can download it, and sends it to the buyer in chat.

### Why it is useful
- Agents today send many property-detail PDFs. They take time to prepare and longer for the buyer to read.
- One short related-property video replaces those PDFs, so the buyer understands the projects faster.
- It speeds up and clarifies agent-buyer communication.
- It is built from the buyer's real searches, not generic marketing.

### Expected business outcomes
- Higher engagement on lead follow-ups and faster agent-to-buyer communication.
- Better conversion of AI-qualified leads into meetings and deals.
- A differentiated agent experience compared with text and PDF packs.
- A reusable asset: the agent can download, resend or share it.

---

## 2. How the decisions evolved

| # | Decision | Detail |
|---|---|---|
| 1 | Location | Agent chat, in the `+` menu next to "Send properties" and "Schedule meeting" |
| 2 | First input idea | Buyer's AI account summary (summary + key insights). This summary is **not stored**; it is generated on demand by the AI API |
| 3 | First length | 10 seconds |
| 4 | Length changed | **60 seconds** |
| 5 | Privacy | No buyer name and no personal intent wording. Only the search details and what the buyer was looking for |
| 6 | Input changed | **Skip the summary.** Use the buyer's actual property and community searches from the database |
| 7 | Agent control | Agent sees an editable input before generating; can filter by bedrooms (1 to 4 or any) and choose projects and communities |
| 8 | Content | Property photos, community and property name, recent sales, recent rents, handover date, with a spoken explanation |
| 9 | Script | **Template-based, no LLM.** Facts are filled from the database. (An LLM was ruled out for the script; the existing ACOT LLM must not be used for it) |
| 10 | Stack | Google Cloud TTS for voice, FFmpeg for video |
| 11 | Backend language | **Python**, inside the Acot-AI- repo |
| 12 | Deployment | Runs on a **VM**, as a separate uvicorn process from the chat AI |
| 13 | Data source | `acot.messages` to choose projects, `propai_*` tables for the facts |

Generative video models (Veo, Kling, Runway, Wan) were considered and rejected as the core: they cost much more, and they can distort real property photos and facts. Veo 3.1 Lite is also video-only, so it would still need voice.

---

## 3. User flow

```text
Agent chat -> + menu -> Generate video
  -> input box (pre-filled from the buyer's searched projects and communities)
  -> agent edits: projects, communities, bedrooms (1-4 or any)   [optional: enhance]
  -> Send
  -> backend reads the data and builds the 60s video
  -> video is ready: agent previews, downloads, or sends it to the buyer in chat
```

Not in the original sketch, added in the design:
- **Preview** before sending.
- **Download** of the MP4.
- **Progress and failure** states, with retry.
- **Empty state**: a buyer with no property searches cannot get a video.

### The prompt step
A free-text prompt that decides which data is fetched would need an LLM to parse it, and it could pick wrong projects. The agreed design is **structured**:
- Projects found in the buyer's searches, with checkboxes.
- Focus options (price, handover, area, investment) and a bedroom filter.
- An optional extra note that only adds wording to the narration and never changes which data is fetched.

---

## 4. The 60-second video

| Time | Content |
|---|---|
| 0 to 8 s | Intro: the communities and bedroom type searched |
| 8 to 50 s | 3 to 4 projects, about 10 to 15 s each: photos, community and project name, recent sale, recent rent, handover date |
| 50 to 60 s | Wrap-up |

- Narration is about 130 to 150 words from a fixed template, with figures filled in from the database.
- Every sentence is **optional**. If a project has no recent sale, no rent or no handover date (ready projects have none), that line is left out, or says "ready to move in".
- Wording stays factual ("recent sales average AED X"), with no promotional claims.
- The voice is baked into the MP4, so it plays with the video in preview, download and chat.

---

## 5. Data

### Where the data lives
Database `acotdb`, Postgres.

| Need | Table |
|---|---|
| What the buyer searched | `acot.messages` (`response_json`), linked by `acot.conversations.user_id` |
| Project name, community, developer, handover | `public.propai_projects` |
| Photos | `public.propai_project_images` |
| Recent sales | `public.propai_sales` |
| Rents | `public.propai_rents` |
| Sales and rent volume, price stats | `public.propai_project_prices`, `public.propai_area_stats` |
| Payment plans, project status | `public.propai_project_plans`, `public.propai_project_status` |

Path from a lead to their searches:

```text
lead.user_id -> acot.conversations (user_id) -> acot.messages (conversation_id)
  -> role = 'assistant' -> response_json -> data.projects / listings / transactions / communities
```

`acot.messages` holds one row per half-turn: `id, conversation_id, turn_id, client_request_id, role, question, status, error, response_json, created_at`. It currently has 1,520 user rows and 1,520 assistant rows. The question is on the user row and the answer (`response_json`) on the assistant row.

Other chat tables (`agent_lead_messages`, `agent_direct_messages`, `agent_ai_*`) are for other chats and are not used.

### Decision: use two layers
- **`acot.messages`** only to choose which projects the buyer looked at.
- **The `propai_*` tables** for everything shown or spoken in the video, because they are complete, current, and the photos are on ACOT's own bucket.

Reading whole tables on every request would be too slow, so the video reads **by project key**, one project at a time (see section 9).

### What a saved answer contains (measured)
Checked on the latest 1,419 saved assistant answers.

Project items: 4,322.

| Photos in the response | Items |
|---|---|
| 4 | 2,560 (2,534 are 4 different images) |
| 3 | 267 |
| 2 | 278 |
| 1 | 131 |
| 0 | 1,085 (25%) |
| 5 | 1 |

- The response is saved with at most about 4 photos per project.
- **Listing items** (390) carry 1 photo each (389 of them).
- The **chat card shows only one photo** per project: the front end picks the first valid image (`projectPhoto()` in `AcotAnswer.tsx`). The saved data holds more than the card shows.
- Photo hosts in saved responses: `new-projects-media.propertyfinder.com` 6,151, `storage.googleapis.com` (ACOT bucket) 5,567, a Cloudflare R2 link 15.
- Only **1,924 of 4,322** project items have a catalog `key` (45%). The other 2,398 come from the Property Finder side and cannot be matched directly to the `propai_*` tables; for those the saved photos and price are the only data.

### What the market tables contain (measured)

| Table | Rows |
|---|---|
| `propai_projects` | 5,639 |
| `propai_project_images` | 21,104 |
| `propai_sales` (not deleted) | 1,400,734 |
| `propai_rents` (not deleted) | 1,519,340 |
| `propai_project_prices` | 5,248 |
| `propai_project_plans` | 2,118 |
| `propai_project_status` | 3,478 |
| `propai_area_stats` | 247 |

Photos per project in `propai_project_images`:
- 2,750 projects have no photos, 2,889 have at least one.
- Most common is 4 photos (433 projects); some have 10 to 59.
- 3+ photos: 2,604 projects. 4+ photos: 2,248. 6+ photos: 1,519.

Coverage by project (out of 5,639):

| Has | Projects |
|---|---|
| Price stats | 5,277 |
| Handover info | 4,165 |
| Recent sales (any) | 3,929 |
| Rents (any) | 3,410 |
| 4+ photos | 2,248 |
| 4+ photos and sales | 2,167 |
| 4+ photos, sales, rents (and handover) | 940 |

Among the 2,248 projects with 4+ photos, in the last 12 months:
- 3+ sales: 1,990
- 3+ rents: 895
- 3+ sales and 3+ rents: 849

So a full video (photos, sales, rents, handover) is possible for about **940** projects, and **2,167** can have photos plus sales. The template must handle projects with missing rents or handover.

### Project selection rules
1. Start from the projects and communities in the buyer's saved searches, deduplicated by project `key`, ranked by how often they appeared and how recent.
2. Apply the agent's choices (projects, communities, bedrooms 1 to 4).
3. Drop projects with fewer than **4 photos** (a minimum of 3 is the absolute floor) and projects with no recent sales and rents.
4. Rank by recent sales volume, then rent volume.
5. Keep the top **3 to 5**.
6. For each, load 4 or more photos, the latest sales and rents (filtered by bedrooms), and the handover date.
7. Items without a catalog `key` fall back to the photos and price saved in the answer.
8. If fewer than 2 usable projects are found, the button shows "Not enough property searches yet" instead of making a weak video.

If the agent names only a community, the backend takes that community's top projects by the same ranking.

### Data caveats
- Photos from Property Finder links may change or expire, and reusing them in a video sent to buyers is a business or licensing question to settle. ACOT's own bucket photos do not raise that question.
- Prices in saved answers are from the day of the search. The video should re-read prices and status at generation time.
- Sales and rent data per project are only present in saved answers if the buyer asked about them, so they come from the market tables.
- The market tables are large (1.4M sales, 1.5M rents). Per-project lookups need an index on `location_key_project`. That has not yet been checked.

---

## 6. Stack and cost

| Layer | Choice | Why |
|---|---|---|
| Script | Fixed template, no LLM | Cheap, fast, and every fact comes from the database |
| Voice | Google Cloud Text-to-Speech (Neural2 or WaveNet) | Low cost, clear English |
| Video | FFmpeg slideshow | Uses real photos; cents per video |
| Slides | Pillow draws each slide (photo, text bar, name, price, handover); FFmpeg stitches them with a slow zoom and the audio | Avoids FFmpeg `drawtext` quoting and font problems |
| Storage | Google Cloud Storage bucket for MP4s | The photos already live on GCS |
| Data | Postgres `acotdb` | Existing |

### Integration
- Google TTS: an API, using the `google-cloud-texttospeech` package and a service account.
- FFmpeg: not a hosted API. It is a binary installed on the server. `imageio-ffmpeg` can supply one for Windows development.
- Packages to add to `requirements.txt`: `google-cloud-texttospeech`, `google-cloud-storage`, `Pillow`, `imageio-ffmpeg`.
- For deployment, the image needs FFmpeg and a font (`ffmpeg`, `fonts-dejavu-core`).

### Cost (estimates from pricing researched in Nov 2026; check current prices)

| Item | Approx. cost per video |
|---|---|
| Google TTS (about 150 words, about 900 characters) | well under $0.01 (Standard/WaveNet about $4 per 1M characters, Neural2 about $16 per 1M) |
| FFmpeg render | about $0 (VM only) |
| Storage and bandwidth | low (a few MB per video) |
| Typical total | about $0.01 to $0.05 |
| Generative AI video (not used) | about $0.50 to $2 or more |

1,000 videos per month: roughly $10 to $50 for TTS and storage, excluding the VM.

---

## 7. Architecture

Python does all the generation. The Nest backend keeps agent login, lead ownership, rate limits and posting to chat.

```text
Agent UI
  -> Nest backend      (auth, lead -> userId, ownership, rate limit)
  -> Python video svc  (internal key)  history -> pack -> script -> TTS -> render -> upload
        reads:  acot.messages, propai_* tables
        writes: acot.agent_videos (job status), GCS (the MP4)
  -> Nest backend      posts the MP4 link into the lead or direct chat
```

### Module layout (Acot-AI- repo)
The folder exists at `app/Viedo_generation/`. All files are currently empty.

```text
app/Viedo_generation/
  history.py   # acot.messages -> projects and communities the buyer searched
  pack.py      # rank by sales and rents, filter bedrooms, photos, handover
  script.py    # fixed 60s template (no LLM, no buyer name)
  tts.py       # Google Cloud TTS -> mp3
  render.py    # Pillow slides + FFmpeg -> mp4
  storage.py   # upload to GCS
  jobs.py      # job status in the database
  router1.py   # FastAPI router
```

Suggested cleanups: rename the folder to `video_generation` (the current spelling is part of every import path) and `router1.py` to `router.py`.

### Pipeline for one job

```text
POST job -> save job (queued)
  -> history: projects from acot.messages
  -> pack: pick top 3-5, load photos, sales, rents, handover
  -> script: fill the 60s template
  -> tts: mp3
  -> render: slides + FFmpeg -> mp4 (voice included)
  -> storage: upload -> URL
  -> job = ready
```

Slide timing follows the narration length so audio and visuals stay in sync.

### Python endpoints (internal, called only by Nest)
- `POST /api/video/jobs`
- `GET /api/video/jobs/{id}`

### Nest endpoints (agent-facing, proposed)
Under the existing `AgentOtpGuard` and the lead ownership check:

| Method | Path | Purpose |
|---|---|---|
| GET | `/agent/leads/:leadId/video/prompt` | Pre-fill: projects and communities the buyer searched |
| POST | `/agent/leads/:leadId/video` | Create the job |
| GET | `/agent/leads/:leadId/video/:id` | Poll status; returns the URL when ready |
| POST | `/agent/leads/:leadId/video/:id/send` | Post the video into the lead or direct chat |

### Job table `acot.agent_videos` (proposed)
`id, agent_id, lead_id, lead_type, status (queued | rendering | ready | failed), request (jsonb), pack (jsonb), video_url, error, created_at, finished_at`. Nest owns the migration, as it does for other tables.

### Delivery in chat
Chat messages are text. Property cards, meetings and offers are already text in the message `body`, which `ChatMessageBody` turns into cards. The video can work the same way: a message whose body has a short caption and the MP4 link, and a small change so `ChatMessageBody` shows a video player for `.mp4` links. No database change is needed, and SSE and history continue to work. The player must be added for both agent and buyer chat.

---

## 8. Running it

The service runs on a **VM**. The chat AI and the video service run as two separate uvicorn processes at the same time:

```text
uvicorn api:app        --port 8000   # chat AI (existing, unchanged)
uvicorn video_api:app  --port 8100   # video generation (new)
```

- The code lives in the Acot-AI- repo and reuses `app/database/postgres.py` and `.env`. Do not import it into `api.py`, so the chat process never loads FFmpeg.
- A new entry file, `video_api.py`, mounts the router.
- Nest calls the video service through a new `VIDEO_API_URL`, the way it uses `AI_API_URL` for chat.
- Run both under `systemd` or two Docker containers, set to restart on failure.
- The repo's Docker image installs torch and embeddings for chat, so it is heavy. On the VM, a second process from the same checkout avoids deploying a second large image.

### Why a separate process
The chat AI runs one uvicorn worker, so one event loop handles all work. A 60-second render uses a lot of CPU. In the same process it would slow `/api/ask` for every buyer, and a crash could take chat down. A separate process isolates both.

### Running both at once
Both processes read the same database. Reads do not conflict. The video service writes only its own job table.

| Area | Issue | Mitigation |
|---|---|---|
| DB connections | Each process has its own pool (`DB_POOL_MAX` default 10) | Give the video process a small pool, for example 3, and check the database's connection limit |
| DB load | Sales and rents aggregates could compete with chat queries | Fetch only a few projects, run queries one at a time, set a statement timeout, cache each project's data for a few hours |
| CPU | FFmpeg uses a lot for about 30 to 90 seconds | Limit to 1 render at a time, lower priority (`nice`), cap videos per agent |
| Memory | The chat process already holds torch and embeddings | Check VM RAM, keep slides at 1280x720, delete temp files after each job |
| Disk | Photos and the MP4 sit in a temp folder | Use a temp directory outside the repo, clean up in a `finally` block |
| Access | The video port must not be public | Allow only the Nest backend and require an internal key header |

Reading `acot.messages`: select only the user's latest 20 to 30 completed assistant messages, and extract just `data.projects`, `transactions`, `listings` and `communities` in SQL rather than loading whole answers.

---

## 9. Risks

| Risk | Detail | Handling |
|---|---|---|
| Chat cannot carry video yet | Messages are text only | Link in `body` plus a player in `ChatMessageBody` |
| Blocking the AI service | One event loop | Separate process; FFmpeg as a subprocess, DB and TTS calls off the event loop |
| Large tables | 1.4M sales, 1.5M rents | Always query by project key; confirm an index on `location_key_project`; cache |
| Missing data | Some projects lack rents, sales, handover or photos | Optional template sentences; minimum photos; "not enough searches" state |
| Keyless projects | 55% of saved project items have no catalog `key` | Fall back to the saved photos and price, or match by name |
| Photo links | Property Finder links can change; reuse needs a licensing decision | Download at generation time, fall back to another photo, prefer ACOT bucket photos |
| Stale prices | Saved prices are from the search day | Re-read prices and status at generation time |
| Authentication | The AI API has no auth of its own | Internal key on video routes; Nest does ownership checks |
| Rendering time | A 60s video takes a while to render | Progress state, retry, a timeout, render limits |

---

## 10. Status and next steps

### Done
- Feature scoped and the flow agreed (sections 1 to 4).
- Data sources identified and measured (section 5).
- Stack chosen (section 6) and architecture agreed (sections 7 and 8).
- Module folder created at `Acot-AI-/app/Viedo_generation/`. All eight files are empty.

### Not done
- No code in any module, no `video_api.py`, no endpoints, no job table.
- No packages added, no FFmpeg in the Docker image or on the VM.
- No Google Cloud credentials, TTS setup or video bucket.
- No chat video player or agent UI.
- Not checked: whether there is an index on `location_key_project` in the sales and rents tables.

### Build order
1. Smoke script: three photos plus a 15-second TTS line becomes an MP4. This proves the stack works.
2. `history.py` and `pack.py`, tested on real buyers' data.
3. `script.py`, `tts.py` and the full 60s render.
4. `storage.py`, `jobs.py` and the two Python endpoints, plus `video_api.py`.
5. Nest endpoints, the chat video player and the agent UI.

### Open decisions
1. Rename the folder to `video_generation` and `router1.py` to `router.py`.
2. A Google Cloud service account for Text-to-Speech and Storage, and the name of the video bucket.
3. Whether FFmpeg is installed on the VM, or to use `imageio-ffmpeg`.
4. Licensing for Property Finder photos inside generated videos.
5. Whether a finished video can be reused when an agent requests the same projects again.
6. Whether the VM has enough spare CPU and RAM to render next to the chat AI.
