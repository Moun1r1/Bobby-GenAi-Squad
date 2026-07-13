import { AVATAR_BG, AVATAR_FG, hashIdx, initials } from "@/lib/avatars";

export function Avatar({ name, size = 36 }: { name: string; size?: number }) {
  const i = hashIdx(name, AVATAR_BG.length);
  return (
    <span
      className="rounded-full grid place-items-center font-semibold shrink-0"
      style={{ width: size, height: size, background: AVATAR_BG[i], color: AVATAR_FG[i], fontSize: size * 0.36 }}
    >
      {initials(name)}
    </span>
  );
}
