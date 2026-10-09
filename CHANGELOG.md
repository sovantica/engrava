# Changelog

All notable changes to engrava will be documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

## [0.7.1](https://github.com/sovantica/engrava/compare/v0.7.0...v0.7.1) (2026-10-09)

### Fixed

* **docs:** ship the VS Code MCP example VS Code actually reads ([0a2afbb](https://github.com/sovantica/engrava/commit/0a2afbbd73e8c92d32f337d116abd8934ec06953))
* **infra:** build EngravaManager stores with the full configuration ([5ebdc46](https://github.com/sovantica/engrava/commit/5ebdc46d89e6478664f88361148cf6e9994b202f))
* **lifecycle:** keep pinned reflections through the sweep, gc and TTL ([4928047](https://github.com/sovantica/engrava/commit/4928047095138216edabb137a7543e5f5ac17993))
* **mindql:** extension handlers get a read-only accessor instead of the live connection ([847766a](https://github.com/sovantica/engrava/commit/847766a15bb3afebe28575144bc0e7b089982d6d))
* **mindql:** read values ending in or/and, and quoted literals, correctly ([c5e7842](https://github.com/sovantica/engrava/commit/c5e7842b2bdfab4eb12c7fdaf2f78c311a9baaac))
* **mindql:** the single-SELECT guard reads SQL comments and quoted identifiers ([b34fcdf](https://github.com/sovantica/engrava/commit/b34fcdf22914ab55db2082f45524096c47972039))
* **search:** raise the per-arm candidate pool to at least top_k ([3a34d0e](https://github.com/sovantica/engrava/commit/3a34d0e33eb35b3bcdd0d12c292c342ecfbf322d))
* **search:** single-source the hybrid-search defaults on SearchConfig ([424efe2](https://github.com/sovantica/engrava/commit/424efe22f5a5dc5f58fc5c182de0851a0b109180))

## [0.7.0](https://github.com/sovantica/engrava/compare/v0.6.0...v0.7.0) (2026-10-02)

### Added

* **cli:** add remember, recall and link one-shot commands ([599d58b](https://github.com/sovantica/engrava/commit/599d58b064cb6c879f2f605624c1a4080ff797d6))
* **core:** add a pre-insert preparation seam and stop bypassing overrides ([6ba6660](https://github.com/sovantica/engrava/commit/6ba66603613befdc4f100138aa0f93d1535fce96))
* **core:** add a public way to attach a dreaming extension ([d6f2966](https://github.com/sovantica/engrava/commit/d6f29669f695d345fec73a234fe45e5b0ab19ce0))
* **core:** guard updates with an engine-maintained revision ([809b3f4](https://github.com/sovantica/engrava/commit/809b3f4046f285264ee1263da0ab28d147acaa85))
* **release:** check that main carries the newest version tag ([92f9ec8](https://github.com/sovantica/engrava/commit/92f9ec8466f7210efe682ce8902cb85195b2f4aa))
* **release:** fail a release that moves the core schema on a patch bump ([bc894b8](https://github.com/sovantica/engrava/commit/bc894b856cb1b91565ed775fceedf1447932d04a))
* **release:** fail a release whose computed version disagrees with the target ([984abb6](https://github.com/sovantica/engrava/commit/984abb628e85a7633b7b0c5c482fd58056d94ba6))
* **release:** fail a run that silently published nothing for the target ([10c3d63](https://github.com/sovantica/engrava/commit/10c3d6359c8ffa60a995108e7d23b68078455aec))

### Fixed

* assert atomic rollback, not partial survival, in v20->v21 test ([5854aa5](https://github.com/sovantica/engrava/commit/5854aa528897ca962d437316f6228cedc6ec1dcf))
* clear ruff check findings on release/v0.7.0 (17 -> 0) ([fac993f](https://github.com/sovantica/engrava/commit/fac993f3e19d2ae2f386995423251ff583ac1a58))
* **cli:** check the schema version before recall, remember and link open a database ([255b97b](https://github.com/sovantica/engrava/commit/255b97b9a2f67fc797b15586f95cf33d8fc1cd95))
* **cli:** clear the journal too when restore --clear wipes the store ([dbda8b9](https://github.com/sovantica/engrava/commit/dbda8b91a63cd108fcdf95f7a3b8723a671cd8a6))
* **cli:** error on a corrupt database instead of hanging forever ([814afb7](https://github.com/sovantica/engrava/commit/814afb79393725e8cded25625cd16f70516a321c))
* **cli:** exempt reflection centroid rows from restore's identity lock ([c8c2c22](https://github.com/sovantica/engrava/commit/c8c2c22240a0f21235e78d8d0c8e12da184a9c7f))
* **cli:** keep the full-text index consistent on every restore ([c588b0f](https://github.com/sovantica/engrava/commit/c588b0fdbf9d4ed5415269e7824dd0e2b36b7a45))
* **cli:** make a snapshot export observe one consistent database state ([d6d31dd](https://github.com/sovantica/engrava/commit/d6d31dd345d2e97b5b7b31c1a3e0ae11d7d59e4c))
* **cli:** name the database in the error message ([e048343](https://github.com/sovantica/engrava/commit/e048343bf1c822b86df44f12a6aeb71e18040ff7))
* **cli:** read export's thought and edge tables in one transaction ([281b65f](https://github.com/sovantica/engrava/commit/281b65f25a3158e43bbfe550b1b6af3cd43a97fa))
* **cli:** refresh a persisted vector index entry a merge restore replaces ([1441ab3](https://github.com/sovantica/engrava/commit/1441ab3b6891e1f089b4ce6865962e6f0f9b1f9a))
* **cli:** refuse a colliding merge into a store that has a journal ([b4912bc](https://github.com/sovantica/engrava/commit/b4912bc02fb1a4af99e934b9d007d2816e77071e))
* **cli:** reject a malformed --filter key as a usage error ([aa0a51c](https://github.com/sovantica/engrava/commit/aa0a51ca0b72687e2771c049163fc512bb7be837))
* **cli:** reject a restored centroid whose vector width disagrees ([58d6f0b](https://github.com/sovantica/engrava/commit/58d6f0bd390b5a69eb20ce1713a6c1bb66013c82))
* **cli:** state what gc deletes, and pin the guards that stop it ([0403e2c](https://github.com/sovantica/engrava/commit/0403e2cec5eb4bc15f862d5aeb0ea88939aa9719))
* **cli:** tell the two schema versions apart ([5e5c2c1](https://github.com/sovantica/engrava/commit/5e5c2c10253b15556a907e8172adb78d3fa6dcde))
* **cli:** verify embedding identity against the target, not a provider ([679c5d4](https://github.com/sovantica/engrava/commit/679c5d4e941eb27205c5bfea43f5d1e6e4939398))
* **cli:** write snapshot and export output through a temp file ([e74133b](https://github.com/sovantica/engrava/commit/e74133b4a92e9fad42d2690b399a8aff589eab60))
* **core:** a no-op upsert_by_hash() no longer commits the caller's pending work ([d92823b](https://github.com/sovantica/engrava/commit/d92823befb747048d40ea1f110f9b0aeb9bb6d42))
* **core:** a write waits for another process's lock again instead of failing at once ([356a08a](https://github.com/sovantica/engrava/commit/356a08a9bece15c04a2a90669d3e71768461d308))
* **core:** bound the close of a wedged connection ([92bbef5](https://github.com/sovantica/engrava/commit/92bbef512bf150f4c1c4419a3dafc8464b47ec7d))
* **core:** close the connection even when cancellation lands mid-cleanup ([f7d592e](https://github.com/sovantica/engrava/commit/f7d592e6c0e56fe13d8fb59c59fa2bb901ec9e81))
* **core:** drop a stale auto-embed completion instead of installing it ([494c0dd](https://github.com/sovantica/engrava/commit/494c0ddb7e6dede9763164131ff4b7f062de3bc8))
* **core:** exempt centroid vectors from the embedding identity lock ([a70bcf6](https://github.com/sovantica/engrava/commit/a70bcf65d8bb65cb9dbccef40afae1f85202fbc9))
* **core:** export the exceptions the documentation says to catch ([ac3d892](https://github.com/sovantica/engrava/commit/ac3d892071ac1ace02e79dfbe5374a6c402af926))
* **core:** keep buffered accesses when a flush cannot take the write lock ([cee9673](https://github.com/sovantica/engrava/commit/cee9673d82e553a157cfe28e0d865e1acc6837bb))
* **core:** keep the no-op upsert from committing after the seam merge ([5f8787a](https://github.com/sovantica/engrava/commit/5f8787aed44afe245fd89e452ae1088d81f9ce60))
* **core:** keep the original error when rollback or pragma cleanup fails ([332a318](https://github.com/sovantica/engrava/commit/332a3188eb2bb1edc4d8bbc0821b6cb331d99fd7))
* **core:** keep the update contract under contention, and journal what a delete really removed ([554b354](https://github.com/sovantica/engrava/commit/554b354b89cefb60bc8e41c4c707b0f6fb2fa666))
* **core:** make a read-modify-write a critical section across tasks ([48ff5ae](https://github.com/sovantica/engrava/commit/48ff5ae72c8fa038042b74a13b7688a16dba209b))
* **core:** make a vector's ownership follow its thought, not its row ([807874e](https://github.com/sovantica/engrava/commit/807874e872f791fb4448bbd6e21328be2dae0d63))
* **core:** make a write and its confirming read-back failure-atomic ([af09cf5](https://github.com/sovantica/engrava/commit/af09cf57870681059c776ccd4f6a3c015aed9fcf))
* **core:** make an embedding write and its vector-index update atomic ([a41f02e](https://github.com/sovantica/engrava/commit/a41f02e7d77e94344fb5b337b9b51d23a8fa87b1))
* **core:** make the child deletes atomic with the parent delete again ([33982b6](https://github.com/sovantica/engrava/commit/33982b6889c2f30ba62c268d9748ee227b338a5a))
* **core:** recover a row update and its journal entry together ([aa5acd8](https://github.com/sovantica/engrava/commit/aa5acd865cffb575128ec12948e6951cba422b77))
* **core:** reject a negative or boolean get_edges limit ([d9f2dfb](https://github.com/sovantica/engrava/commit/d9f2dfbbbd38deed347a156bb79e4001008289ef))
* **core:** roll back or quarantine a failed commit instead of leaving it open ([cb1355f](https://github.com/sovantica/engrava/commit/cb1355f7b73a055ca57f8f88ffb06d9a5e1d4c4f))
* **core:** sample transaction ownership again before upsert_by_hash's decisive probe ([7a8fd06](https://github.com/sovantica/engrava/commit/7a8fd063d37a293fc1deb85b75846da86ebfb918))
* **core:** say when the unembedded row survives an auto-embed failure ([059efa4](https://github.com/sovantica/engrava/commit/059efa47989a84598d839c233df803f1fd455212))
* **core:** serialise the dedup probe-and-insert across connections ([7bb7c6c](https://github.com/sovantica/engrava/commit/7bb7c6ca4070c54f6265395c388d2b5ca46a3b9f))
* **core:** skip derivation only inside the current task's own transaction window ([067f28c](https://github.com/sovantica/engrava/commit/067f28caa48f87de3c0a38e557482dbd92745465))
* **core:** stop a write-free call from committing the caller's transaction ([6f622a2](https://github.com/sovantica/engrava/commit/6f622a266a21b40976486b0cdae0227e56d7f22a))
* **core:** store every timestamp in one canonical UTC form ([02cd72d](https://github.com/sovantica/engrava/commit/02cd72d65ca7cb8610f939de134e89c38d639770))
* **docs:** correct auto-embed durability claims in api-reference/configuration ([6b6b628](https://github.com/sovantica/engrava/commit/6b6b628082e517633d8debf28daca87f17f53369))
* **docs:** correct EmbeddingConfig.require_embedding durability docstring ([a2254e5](https://github.com/sovantica/engrava/commit/a2254e53deb8a72705386448235c5d823e94a99e))
* **docs:** split auto-embed durability into two independent questions ([4a77d8e](https://github.com/sovantica/engrava/commit/4a77d8ef19a7085330c5641840262d9bf88e496c))
* **docs:** stop conflating durability with method, add missing precondition ([e1c00fa](https://github.com/sovantica/engrava/commit/e1c00fafc1467a79439ccad202420ea7b32bc967))
* **docs:** teach the cleanup the library ships in every example ([a6c61fa](https://github.com/sovantica/engrava/commit/a6c61fa554ffdc03c255f005bd8bfe6753677d08))
* **domain:** say whether a metrics snapshot was actually measured ([918161c](https://github.com/sovantica/engrava/commit/918161c20bb5090c653627b47cf15d020faea0a5))
* **dreaming:** compute the cohesion gate as the cosine similarity it names ([a851882](https://github.com/sovantica/engrava/commit/a851882c6628844fd63138dc04bc657de0f5f7de))
* **dreaming:** roll back a reflection whose own embedding fails ([2fde4af](https://github.com/sovantica/engrava/commit/2fde4afe0b1a3e785ba3a179a3fa2afff292288e))
* **embeddings:** stop a rejected first write from locking corpus identity ([6f2dbc9](https://github.com/sovantica/engrava/commit/6f2dbc9cc982675302704654b14a9f06c8aff556))
* **embeddings:** stop logging the embedding provider's raw error message ([06b323d](https://github.com/sovantica/engrava/commit/06b323d437e5c017f12bb452c4b8f399b04bb456))
* **extensions:** roll back the migration savepoint on cancellation too ([c7d9708](https://github.com/sovantica/engrava/commit/c7d9708f68f5bf1a26786476238df466a1103147))
* **hygiene:** make a failed or cancelled hygiene pass leave nothing behind ([bbb1ca2](https://github.com/sovantica/engrava/commit/bbb1ca277c17c0d88a829fcd0cf816771f0df64d))
* **infra:** keep one manager caller's cancellation away from the others ([60acdac](https://github.com/sovantica/engrava/commit/60acdac858b899115d15804304ee5ddc3932925b))
* **infra:** share one lifecycle lock across create, delete, and close_all ([1a157f6](https://github.com/sovantica/engrava/commit/1a157f670e7eb2a312bed027759d2806779b685c))
* **infra:** wait for the aiosqlite worker after a failed connect ([203baf9](https://github.com/sovantica/engrava/commit/203baf90d37f835af192a5289f1592f03b426df1))
* **journal:** a failed derived record undoes only itself, even inside your transaction ([7aebc9a](https://github.com/sovantica/engrava/commit/7aebc9aabcc83f9083a2d3d4532a3b64eeb50c1e))
* **journal:** make create, delete and outcome writes recover together with their journal entry ([68eb360](https://github.com/sovantica/engrava/commit/68eb36081b8adaa478113d0a8d7a3854e813de8f))
* **migration:** make a migration step apply completely or not at all ([2f5a949](https://github.com/sovantica/engrava/commit/2f5a94905af63dff92f271aa7f4fdf11a1f8a063))
* **migration:** roll back legacy-history adoption's savepoint on cancellation too ([304327c](https://github.com/sovantica/engrava/commit/304327c7e4405aae567ebb7dd4d2cab2e42a327b))
* **migration:** unwind a savepoint cancelled while it opens ([0324c1f](https://github.com/sovantica/engrava/commit/0324c1f60f07802a58fcc4d7eb97fffb12b09d35))
* **mindql:** match quoted values exactly, including repeated whitespace ([481fbee](https://github.com/sovantica/engrava/commit/481fbee97a7595988f58cc93b62950c8134fb75f))
* **mindql:** reject a LIMIT or OFFSET too large for SQLite to store ([7ce08ce](https://github.com/sovantica/engrava/commit/7ce08cefdac321cd682f7b1d6d848c1e484692c9))
* **release:** close the gaps the last round's claims papered over ([a71edc8](https://github.com/sovantica/engrava/commit/a71edc8f52b1ccfaabef07e9971f3a3d29ada8b0))
* **release:** one boundary around reading the target, not a longer catch list ([e5b46e9](https://github.com/sovantica/engrava/commit/e5b46e9fcb2b38e213835cdec19e32a2e7262e16))
* **release:** one lookup, one claim, and close three more false passes ([48ebba7](https://github.com/sovantica/engrava/commit/48ebba70a2e71e11ae630f41750380d4d9dbb385))
* **release:** stop the was-published gate from passing on nothing ([bdeb668](https://github.com/sovantica/engrava/commit/bdeb668d8ca9fa06b62ae4e7d4cbd1c1cebf4a33))
* **search:** apply collapse and reflection caps to the query-less fallback ([1543c7f](https://github.com/sovantica/engrava/commit/1543c7fb76893e468bfc53a91382949b454a2647))
* **search:** bound edge retrieval and graph ranking's neighbour fetch in SQL ([60f7b47](https://github.com/sovantica/engrava/commit/60f7b475baca4fdda75aa640c8066e6360d4c1ca))
* **search:** keep search query text out of the FTS fallback logs ([1db1b6c](https://github.com/sovantica/engrava/commit/1db1b6c050fa9abc0fade4de53e96fdbe2ab8cae))
* **search:** keep the query-less fallback's order under the reflection cap ([af8f162](https://github.com/sovantica/engrava/commit/af8f162de34f0b0cfa8066f7ef4c1cfb05671e2e))
* **search:** stop graph ranking from double-counting a cross-chunk edge ([4b2a0dd](https://github.com/sovantica/engrava/commit/4b2a0dd9e159ba482de5d13a68a29958ac293185))
* **search:** turn cycle recency fully off at a zero weight ([ef1f87a](https://github.com/sovantica/engrava/commit/ef1f87aec738e438e2bcd3bd130a0ed963ba55cf))
* update stale schema-version golden to 21 ([4a78102](https://github.com/sovantica/engrava/commit/4a78102aea48ea24887e02a9cd924e2e3f74b7b2))

## [0.6.0](https://github.com/sovantica/engrava/compare/v0.5.0...v0.6.0) (2026-08-10)

### Added

* add derived-records extension seam ([c9d55cb](https://github.com/sovantica/engrava/commit/c9d55cb1cb6dbcbd980078994b59819f07daacca))
* **core:** add derive_existing() backfill for the derived-records seam ([1c79611](https://github.com/sovantica/engrava/commit/1c79611490a179b9b141ca685a48e76c22a4c3b9))
* **core:** add opt-in cycle-provider seam and max_cycle accessor ([29d46e5](https://github.com/sovantica/engrava/commit/29d46e58afcbd371229654aeba9605ff3f6f88ea))
* **edges:** add generic metadata carrier with schema v19 migration ([08a4cb3](https://github.com/sovantica/engrava/commit/08a4cb3c9923f304bce3011050c95846598308a1))
* **extensions:** add zero-dependency split modes to StructuralSplitProducer ([320b88e](https://github.com/sovantica/engrava/commit/320b88e0cb9f66981f51abd5b48c4f389167ead8))
* **hygiene:** add a wall-clock restore window before permanent GC ([35c2769](https://github.com/sovantica/engrava/commit/35c276997a47b1091e56a62425132329c8e5eca9))
* **hygiene:** guard archival behind a minimum inactivity age and a usage-signal gate ([0ce64b7](https://github.com/sovantica/engrava/commit/0ce64b74e91d3ee270c9ecb4d480d6b621f7491d))
* **search:** add transaction-time recency axis with caller-supplied now ([49f8c9b](https://github.com/sovantica/engrava/commit/49f8c9b8141bbff07f4fb6a7bbfb377d5f069abe))
* **search:** exclude archived thoughts from default retrieval ([963e60c](https://github.com/sovantica/engrava/commit/963e60c32e222f52f2d05d9eb443fc2e4d214e87))
* **search:** reject wrong-dimension query vectors and count vector-arm degradation ([fe128f1](https://github.com/sovantica/engrava/commit/fe128f1161309d700ccb6e3860daaac489d521e4))

### Fixed

* **cli:** cover the expiry sweep and report an unreadable snapshot ([4ab31e4](https://github.com/sovantica/engrava/commit/4ab31e4a01aa174ecb1432d152ae22342d79c304))
* **cli:** give an invalid --service name a distinct, clean error ([5f5372d](https://github.com/sovantica/engrava/commit/5f5372d4408b7836ce72004e4f824e0d60cd2556))
* **cli:** purge the vector index when gc collects a thought ([44c6ffc](https://github.com/sovantica/engrava/commit/44c6ffc956340387b675a431458c7863bf1106a3))
* **cli:** validate a resolved empty or default service name ([6318f98](https://github.com/sovantica/engrava/commit/6318f98c564bd20a3cc1cf40be45fdc450aeedc1))
* **cli:** validate snapshot-restore input against a typed model and restore atomically ([d35dc5b](https://github.com/sovantica/engrava/commit/d35dc5b4bd56635f766f10198b7cbfbb484ba679))
* close Free audit source follow-ups ([d919d85](https://github.com/sovantica/engrava/commit/d919d85e43de1eaae5fce953b684446aa97f58fd))
* **config:** enforce uniform validation across sections and construction paths ([97a9af7](https://github.com/sovantica/engrava/commit/97a9af7d0ddd31fe050e033bea5b7a23c4ffa2b5))
* **config:** make the validated value the value that gets used ([e1d7d00](https://github.com/sovantica/engrava/commit/e1d7d003b4cb5bbebdc0d2a2e1801fdd1378d56f))
* **config:** use assign-to-variable message in unknown-key rejection ([69bce4e](https://github.com/sovantica/engrava/commit/69bce4e0b205cfa36c040def917296b573775097))
* **core:** reject inverted valid_from/valid_until intervals ([dd9305f](https://github.com/sovantica/engrava/commit/dd9305febd4e2d85a4c4c06f0f90d1fc504fa6b0))
* **dreaming:** propagate real integrity failures during edge creation ([79241c2](https://github.com/sovantica/engrava/commit/79241c239f06f17a12430fa8097d467b20fad9a7))
* **embeddings:** name the provider member a search needs instead of failing on it ([5e461f6](https://github.com/sovantica/engrava/commit/5e461f65ae905c2ba2c56947834e41e4c5140790))
* **infra:** harden core bootstrap and edge integrity classification ([be054f6](https://github.com/sovantica/engrava/commit/be054f665422580a032dc4b928d61ca73cd14add))
* **infra:** keep foreign-key enforcement safe on every swap failure path ([c393980](https://github.com/sovantica/engrava/commit/c39398044774293c6ce1ac34ce3bd74a656571b7))
* **infra:** make the v11->v12 child-table swap atomic via a savepoint ([7b9a1d0](https://github.com/sovantica/engrava/commit/7b9a1d0603d69c99912569b0ca2c229952225169))
* **infra:** write only the fields an update owns ([9d72125](https://github.com/sovantica/engrava/commit/9d721252d6408b6fc4e4eb64af1b05c28079dba4))
* **journal:** reclaim per-connection append locks with a weak-key registry ([412dcd8](https://github.com/sovantica/engrava/commit/412dcd8cca67bcff33fd3764cc765b0371fb0840))
* **mindql:** build the passthrough guard on a value the module owns ([6e8db02](https://github.com/sovantica/engrava/commit/6e8db02c1b98b8f30a76d205f016e114f836edb8))
* **mindql:** validate identifiers where the query is executed ([d4d8111](https://github.com/sovantica/engrava/commit/d4d811130cd1e0f3bd57ab2707164fc930a10be6))
* **read-only:** capability-separate the read-only view from the core protocol ([1033a2e](https://github.com/sovantica/engrava/commit/1033a2e8932d92d2944e305d819a1d46a0cd64cd))
* **search:** keep FTS5 MATCH valid so quoted and wildcard queries never silently drop BM25 ([5c99b88](https://github.com/sovantica/engrava/commit/5c99b88a5ec4582c099ed9e767b48b90038a65a0))
* **search:** quote exposed FTS5 boolean operators so bare queries never lose BM25 ([ad7ec85](https://github.com/sovantica/engrava/commit/ad7ec8583e68ed4cba530aca3de31e6f2383302d))
* **sqlite:** harden core migration registry ([5c9ec91](https://github.com/sovantica/engrava/commit/5c9ec9189f27f3a9a7996560ac839af66cae15ee))
* **sqlite:** harden extension migration identity and statement splitting ([cfc5e95](https://github.com/sovantica/engrava/commit/cfc5e9533c869ab96e2c5b7dc701de6eaa1049dc))

## [0.5.0](https://github.com/sovantica/engrava/compare/v0.4.0...v0.5.0) (2026-07-08)

### ⚠ BREAKING CHANGES

* **changelog:** the in-tree MCP server is removed from engrava. The engrava[mcp]
optional-dependency extra, the in-engrava engrava-mcp console script, and the
in-tree server module are gone; a plain 'pip install engrava' is unaffected. The
server moved to the standalone engrava-mcp package (uvx engrava-mcp), which
consumes engrava's public API. Migrate per the docs/upgrade.md 0.4 -> 0.5 notes.

### Added

* **core:** action-outcome feedback loop and mutable action lifecycle ([2008a3f](https://github.com/sovantica/engrava/commit/2008a3f73b474cdf644585ac44c0855b9e0dc21a))
* **core:** add opt-in deterministic memory-hygiene forgetting loop ([9167948](https://github.com/sovantica/engrava/commit/9167948f141cbf3b4dae26a87dfd4d17b96701f5))
* **core:** batch/get-or-create write primitives and embed-failure visibility ([e9c2021](https://github.com/sovantica/engrava/commit/e9c202199c248ea1b0dccde6c89e32ebab39bd03))
* **core:** opt-in typed provenance-context capture at create_thought ([b8b8fa3](https://github.com/sovantica/engrava/commit/b8b8fa3a69ddf062b49882578ebb25f126dd4d83))
* **dreaming:** activate consolidation — reachable scoring + live access substrate ([5114f81](https://github.com/sovantica/engrava/commit/5114f81e6f0b485c8b580da0e3018fe1ebc27ea7))
* **embeddings:** opt-in asymmetric query/document prefixes ([12e61fb](https://github.com/sovantica/engrava/commit/12e61fbd0b8c1d5897fdf7f9186f53336a6cb85e))
* **journal:** expose hash-chain verification via API, CLI, and on-open gate ([b26f519](https://github.com/sovantica/engrava/commit/b26f5191f6dac0985a552c6333e788186e6f3cf7))
* **mindql:** read-surface ergonomics — IN, boolean WHERE, ORDER BY, OFFSET, EXPLAIN, bound SELECT ([eca5a74](https://github.com/sovantica/engrava/commit/eca5a743017b07c44caebbc23ae6994c0ec2c7fb))
* remove the in-tree MCP server (now the standalone engrava-mcp package) ([1417e91](https://github.com/sovantica/engrava/commit/1417e91680298c650b8454e44e948057fa17cc30))
* **search:** add collapse_key de-fragmentation to hybrid retrieval ([6c80277](https://github.com/sovantica/engrava/commit/6c8027759a5073d96a9d73578af5980203efe4a8))
* **search:** add metadata and visibility filters to ranked retrieval ([a56fcc7](https://github.com/sovantica/engrava/commit/a56fcc7bdceb792ea714aaca7e8f447381f5efc6))
* **search:** add opt-in per-unit retention depth for collapse backfill ([e72cd22](https://github.com/sovantica/engrava/commit/e72cd22f5ca2f12724ddd6cdbe70502153f84381))
* **search:** batch read-path decode, inbound edge index, eviction visibility ([0e1b81f](https://github.com/sovantica/engrava/commit/0e1b81f4701ae362c3c49b9481914cc049560dcd))

### Fixed

* **dreaming:** add opt-in cold-start clustering fallback ([35aedab](https://github.com/sovantica/engrava/commit/35aedabe843c43ef7f4c334335c8593cf88c32e4))
* **lifecycle:** make archived thoughts restorable to ACTIVE ([88c2e3e](https://github.com/sovantica/engrava/commit/88c2e3eb3c5bc7a9baaffb513b7e46627a6f5a13))
* **search:** apply filters/visibility in the query-less fallback arm ([2f0640e](https://github.com/sovantica/engrava/commit/2f0640ee76d0947fcbd5708d5f67dbf8709c5c34))
* **search:** correct the sqlite-vec backend — purge deleted vectors, fill top_k ([d884052](https://github.com/sovantica/engrava/commit/d884052e871ddcc4d576471ffcee12cb6fee06ef))
* **search:** neutral midpoint for the degenerate min-max fusion case ([815f23b](https://github.com/sovantica/engrava/commit/815f23bcbeff1b45ffaccf82f7a289e06ca4837b))
* use absolute GitHub links in README so PyPI does not 404 ([acf56bf](https://github.com/sovantica/engrava/commit/acf56bfce59eafa9101295dec753cad7deb356d8))

### Documentation

* **changelog:** curate 0.5.0 Unreleased block and mark MCP removal breaking ([5cf3a78](https://github.com/sovantica/engrava/commit/5cf3a78f2108b84836d1b1a0c1b72a82450f1241))

## [0.4.0](https://github.com/sovantica/engrava/compare/v0.3.1...v0.4.0) (2026-06-18)

### Added

* add bi-temporal valid-time to thoughts and edges ([456bcb6](https://github.com/sovantica/engrava/commit/456bcb6f66710ce9a51a5761dd7e406d15e1d177))
* add temporal query predicates and invalidate primitive ([86de77f](https://github.com/sovantica/engrava/commit/86de77fc11ab4322927648d8c7a51ea6d13da91a))
* **api:** add remember() and recall() convenience methods on the store ([be9b110](https://github.com/sovantica/engrava/commit/be9b110d24494f3641bd504a00c2fc838b55c867))
* **mcp:** add delete_thought and delete_edge tools ([84b87f6](https://github.com/sovantica/engrava/commit/84b87f6569958052605a87fbed1aebb3fe734043))
* **mcp:** add guided memory prompts ([1f6fa36](https://github.com/sovantica/engrava/commit/1f6fa36d1b1fef6d63fa2f949c6f18441c84f25b))
* **mcp:** add MCP server with read tools (engrava[mcp] extra) ([68a8085](https://github.com/sovantica/engrava/commit/68a8085c601eb6343ee21622ff422f0234cdc294))
* **mcp:** add memory filters and pagination ([57357b7](https://github.com/sovantica/engrava/commit/57357b711b584e4b3b5a2d206344d33230f35a9e))
* **mcp:** add write tools, opt-in read-only mode, and per-tool safety annotations ([79d7604](https://github.com/sovantica/engrava/commit/79d7604aa492aeab8b2e8dacbbaab1231738d5c7))
* **mcp:** expose memory as resources (thought, stats, recent) ([c54dcf7](https://github.com/sovantica/engrava/commit/c54dcf77180a31c7bcb06d815de316ae1c93d488))
* **mcp:** map known failures to typed, actionable tool errors ([8b615cc](https://github.com/sovantica/engrava/commit/8b615cc58c5b81adf33a29de16be8060f2bf9bfe))
* **mindql:** add store-level execute_mindql entry point ([1c8ffb4](https://github.com/sovantica/engrava/commit/1c8ffb4b18c3ed817f6e229744e943262608413c))
* reflections inherit temporal extent from members ([8fba769](https://github.com/sovantica/engrava/commit/8fba76984c59e5c9424d68a4bba21be53b72d9a6))

### Fixed

* assert plan-shape invariant for temporal queries, not scan-vs-index ([0e4e176](https://github.com/sovantica/engrava/commit/0e4e1764ac770e837012852099c90c7c26b0c4a4))
* embed full thought content without duplication or silent truncation ([36e08e7](https://github.com/sovantica/engrava/commit/36e08e773f6a48d88b9875e1e42ce7051640e7f9))
* **embeddings:** retry transient errors with bounded backoff ([897c46f](https://github.com/sovantica/engrava/commit/897c46fd9300c4b8b9530852d414043593ffffbe))
* keep quoted MindQL values as strings and reject malformed conditions ([d88043d](https://github.com/sovantica/engrava/commit/d88043df45ab05047c20517a5882f071ad7cab92))
* let natural-language queries reach the full-text index ([bb6b729](https://github.com/sovantica/engrava/commit/bb6b7290d546185c7d01789e204338482e39e7cf))
* match exact table token in query-plan helpers ([cd4ecc2](https://github.com/sovantica/engrava/commit/cd4ecc264dff0e425a0dc0708deb5b68f83fab52))
* **mcp:** keep query_memory parse errors FIND-only ([5f4ea20](https://github.com/sovantica/engrava/commit/5f4ea2009d4e4527c3af1fc429055c61f38e509c))
* **mcp:** map write-tool errors and complete the 0.4.0 documentation ([6794436](https://github.com/sovantica/engrava/commit/6794436cf4eec8e16e01a67091d00435b29d1ca5))

### Changed

* tune sqlite pragmas and add hot-path indexes ([1256303](https://github.com/sovantica/engrava/commit/12563030ebab86c481fae341eaccabd8db2223eb))

## [0.3.1](https://github.com/sovantica/engrava/compare/v0.3.0...v0.3.1) (2026-06-02)

### Fixed

* **vector:** load sqlite-vec extension on the connection's worker thread ([457d2f7](https://github.com/sovantica/engrava/commit/457d2f724f877e74509ca711a122293a151e3b01))
* **vector:** re-disable extension loading in finally after load attempt ([e9ab267](https://github.com/sovantica/engrava/commit/e9ab2679c51508f1c2e0ec350c1635bbf2906be7))

## [0.3.0](https://github.com/sovantica/engrava/compare/v0.2.0...v0.3.0) (2026-06-02)

### Added

* graph memory database — dreaming consolidation, hybrid search, audit trail ([ed82259](https://github.com/sovantica/engrava/commit/ed822599f6e2922ec5b3eed8dc14e4a7bb9b024a))

## [0.2.0] — 2026-04-12

### Breaking Changes

- None.

### Database Changes

- Schema version bumped to core-5.
- Added `access_count`, `last_accessed_at`, `confirmation_count`,
  `consolidated_from`, and `visibility` to the thought table.
- Existing databases upgrade automatically through `ensure_schema()`.

### Added

- **Full-text search (FTS5)** — `search_fts()` method with BM25 ranking on `essence`
  and `content` fields. Hybrid search combines vector similarity, text relevance, and
  recency scoring via configurable `SearchConfig` weights.
- **Extension system** — `EngravaHooksProtocol` with 5 hook points (`on_store`,
  `on_retrieve`, `score_function`, `decay_function`, `mindql_extension_registry`).
  `DefaultEngravaHooks` provides no-op defaults.
- **Dreaming / memory consolidation** — `DreamingExtension` with 5 pluggable signal
  types (`ConfidenceSignal`, `ConfirmationSignal`, `FrequencySignal`, `RecencySignal`,
  `StalenessSignal`) and configurable gate thresholds.
- **sqlite-vec backend** — optional `SqliteVecSearchBackend` for hardware-accelerated
  vector search via the `vec` extra.
- **Multi-service isolation** — `EngravaManager` for running multiple independent
  databases, each with its own schema, embeddings, and FTS index.
- **YAML configuration** — `load_config()` factory, `EngravaConfig` with `SearchConfig`,
  `DreamingConfig`, `EmbeddingConfig`, and `ServicesConfig` sections.
- **5 embedding providers** — `SentenceTransformerProvider`, `OpenAICompatibleProvider`,
  `OllamaProvider`, `HuggingFaceProvider`, `CallbackProvider`.
- **MindQL enhancements** — `COUNT`, `SELECT` with `WHERE` clauses, extensible command
  registry via hooks.
- **CLI enhancements** — `export`, `import`, `gc`, `migrate` subcommands. Multi-service
  `--service` flag for `snapshot`/`restore`/`export`.
- **Read-only store** — `ReadOnlyEngrava` wrapper that raises `ReadOnlyViolationError`
  on write attempts.
- **`ExtensionManifest`** value object for extension discovery and registration.
- **`HybridSearchResult`** model combining vector, FTS, and recency scores.
- **`EmbeddingModelMismatchError`** exception for restore-time model validation.
- Open source release: standalone repository, MIT license, GitHub Actions CI/CD,
  PyPI publishing.

### Changed

- Bumped version from 0.1.0 to 0.2.0.
- `pyproject.toml` URLs now point to the standalone GitHub repository.
- Description updated to "Thought-graph database for AI agents".

### Fixed

- `--re-embed` flag now raises an error when no embedding provider is configured
  instead of silently succeeding.
- `snapshot --service` validates service existence before attempting export.
- `EngravaManager.get_store()` uses `asyncio.Lock` to prevent race conditions
  during concurrent lazy initialization.

## [0.1.0] — 2026-04-01

### Breaking Changes

- Initial release.

### Database Changes

- Initial SQLite schema introduced.
- No downgrade guarantees; follow forward-only upgrade policy from later releases.

### Added

- Initial release (internal).
- `SqliteEngravaCore` — async thought/edge/embedding/action CRUD.
- `ThoughtRecord`, `EdgeRecord`, `EmbeddingRecord`, `ActionRecord` frozen Pydantic models.
- 9 domain enums (`ThoughtType`, `Priority`, `LifecycleStatus`, `EdgeType`, etc.).
- Brute-force cosine similarity embedding search.
- `MindQLParser` and `MindQLExecutor` — `FIND` and basic query support.
- CLI with `info`, `query`, `snapshot`, `restore` subcommands.
- Schema migration support via `ensure_schema()`.
