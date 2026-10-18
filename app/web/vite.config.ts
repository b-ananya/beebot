import react from '@vitejs/plugin-react'
import fs from 'node:fs'
import type { IncomingMessage, ServerResponse } from 'node:http'
import path from 'node:path'
import { defineConfig, type Plugin, type PreviewServer, type ViteDevServer } from 'vite'
import { attachApis } from './server/apis.ts'
import { attachSimApi } from './server/sim-api.ts'

const FILE_MOUNTS: Array<[prefix: string, dir: string]> = [
  ['/clips/', 'clips'],
  ['/tracks/', 'tracks'],
  ['/search/', 'search'],
  ['/field/', 'field'],
  ['/plan/', 'plan'],
  ['/truth/', 'truth'],
  ['/eval/', 'eval'],
  ['/forecast/', 'forecast'],
  ['/rollouts/', 'rollouts'],
]

function contentType(file: string): string {
  switch (path.extname(file).toLowerCase()) {
    case '.mp4':
      return 'video/mp4'
    case '.json':
      return 'application/json'
    case '.webm':
      return 'video/webm'
    case '.jpg':
    case '.jpeg':
      return 'image/jpeg'
    case '.png':
      return 'image/png'
    case '.gif':
      return 'image/gif'
    default:
      return 'application/octet-stream'
  }
}

function sendFile(req: IncomingMessage, res: ServerResponse, file: string, total: number): void {
  res.setHeader('Accept-Ranges', 'bytes')
  res.setHeader('Content-Type', contentType(file))
  res.setHeader('Cache-Control', 'no-cache')

  if (req.method === 'HEAD') {
    res.statusCode = 200
    res.setHeader('Content-Length', total)
    res.end()
    return
  }

  const range = req.headers.range
  let start = 0
  let end = total - 1

  if (range) {
    const match = /^bytes=(\d*)-(\d*)$/.exec(range)
    if (!match) {
      res.statusCode = 416
      res.setHeader('Content-Range', `bytes */${total}`)
      res.end()
      return
    }
    const from = match[1] === '' ? NaN : Number(match[1])
    const to = match[2] === '' ? NaN : Number(match[2])
    if (Number.isNaN(from)) {
      const suffix = to
      if (!Number.isFinite(suffix) || suffix <= 0) {
        res.statusCode = 416
        res.setHeader('Content-Range', `bytes */${total}`)
        res.end()
        return
      }
      start = Math.max(0, total - suffix)
      end = total - 1
    } else {
      start = from
      end = Number.isNaN(to) ? total - 1 : to
    }
    if (start < 0 || start >= total || end < start) {
      res.statusCode = 416
      res.setHeader('Content-Range', `bytes */${total}`)
      res.end()
      return
    }
    end = Math.min(end, total - 1)
    res.statusCode = 206
    res.setHeader('Content-Range', `bytes ${start}-${end}/${total}`)
    res.setHeader('Content-Length', end - start + 1)
  } else {
    res.statusCode = 200
    res.setHeader('Content-Length', total)
  }

  const stream = fs.createReadStream(file, { start, end })
  const stop = () => stream.destroy()
  res.on('close', stop)
  stream.on('error', () => {
    res.off('close', stop)
    if (!res.headersSent) {
      res.statusCode = 500
      res.end()
    } else {
      res.destroy()
    }
  })
  stream.pipe(res)
}

function mountFiles(prefix: string, rootDir: string) {
  const root = path.resolve(rootDir)
  return (req: IncomingMessage, res: ServerResponse, next: (err?: unknown) => void) => {
    const raw = req.url?.split('?')[0] ?? ''
    if (!raw.startsWith(prefix)) {
      next()
      return
    }
    if (req.method !== 'GET' && req.method !== 'HEAD') {
      next()
      return
    }

    let rel: string
    try {
      rel = decodeURIComponent(raw.slice(prefix.length))
    } catch {
      res.statusCode = 400
      res.end()
      return
    }
    if (!rel || rel.includes('\0')) {
      res.statusCode = 400
      res.end()
      return
    }

    const file = path.resolve(root, rel)
    const fromRoot = path.relative(root, file)
    if (fromRoot.startsWith('..') || path.isAbsolute(fromRoot)) {
      res.statusCode = 403
      res.end()
      return
    }

    fs.stat(file, (err, stat) => {
      if (err || !stat.isFile()) {
        next()
        return
      }
      sendFile(req, res, file, stat.size)
    })
  }
}

function mountRepo(server: ViteDevServer | PreviewServer): void {
  const repo = path.resolve(server.config.root, '../..')
  for (const [prefix, dir] of FILE_MOUNTS) {
    server.middlewares.use(mountFiles(prefix, path.join(repo, dir)))
  }
  attachSimApi(server.middlewares, repo) // before attachApis, which answers every other /api/ path
  attachApis(server.middlewares, repo)
}

function clipsPlugin(): Plugin {
  return {
    name: 'beebot-clips',
    configureServer: mountRepo,
    configurePreviewServer: mountRepo,
  }
}

export default defineConfig({
  plugins: [react(), clipsPlugin()],
  build: {
    rollupOptions: { input: { main: 'index.html', sim: 'sim.html' } },
  },
  server: {
    port: 5173,
    strictPort: false,
  },
})
