/**
 * Solutions travel in the URL hash as `#<level id>/<code>`, where the code is
 * deflate-compressed and base64url-encoded, so nothing leaves the browser.
 */

async function pipe(bytes: Uint8Array, stream: CompressionStream | DecompressionStream): Promise<Uint8Array> {
  const out = new Response(new Blob([bytes as BlobPart]).stream().pipeThrough(stream));
  return new Uint8Array(await out.arrayBuffer());
}

function toBase64Url(bytes: Uint8Array): string {
  let s = '';
  for (const b of bytes) s += String.fromCharCode(b);
  return btoa(s).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

function fromBase64Url(text: string): Uint8Array {
  const s = atob(text.replace(/-/g, '+').replace(/_/g, '/'));
  return Uint8Array.from(s, (c) => c.charCodeAt(0));
}

export async function encodeShare(levelId: string, code: string): Promise<string> {
  const packed = await pipe(new TextEncoder().encode(code), new CompressionStream('deflate-raw'));
  return `${levelId}/${toBase64Url(packed)}`;
}

export async function decodeShare(hash: string): Promise<{ levelId: string; code: string } | null> {
  const m = /^#?([a-z0-9-]+)\/([A-Za-z0-9_-]+)$/.exec(hash);
  if (!m) return null;
  try {
    const bytes = await pipe(fromBase64Url(m[2]), new DecompressionStream('deflate-raw'));
    return { levelId: m[1], code: new TextDecoder().decode(bytes) };
  } catch {
    return null;
  }
}
