#!/usr/bin/env python3
"""
NeuroGym - AI-Powered Fitness & Rehab Coach
Main entry point for the real-time vision pipeline.

Usage:
    python main.py                    # Default: all features enabled
    python main.py --no-face          # Disable face tracking for better FPS
    python main.py --no-hands         # Disable hand tracking
    python main.py --camera 1         # Use camera index 1
"""

import argparse
import time
import cv2

from neurogym import VideoProcessor, HolisticDetector, CyberpunkDrawer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="NeuroGym - AI-Powered Fitness & Rehab Coach Vision Pipeline"
    )
    
    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=0,
        help="Camera index (default: 0)"
    )
    
    parser.add_argument(
        "--width", "-W",
        type=int,
        default=1280,
        help="Frame width (default: 1280)"
    )
    
    parser.add_argument(
        "--height", "-H",
        type=int,
        default=720,
        help="Frame height (default: 720)"
    )
    
    parser.add_argument(
        "--no-pose",
        action="store_true",
        help="Disable pose detection"
    )
    
    parser.add_argument(
        "--no-hands",
        action="store_true",
        help="Disable hand detection"
    )
    
    parser.add_argument(
        "--no-face",
        action="store_true",
        help="Disable face detection (improves FPS)"
    )
    
    parser.add_argument(
        "--no-glow",
        action="store_true",
        help="Disable glow effect on landmarks"
    )
    
    parser.add_argument(
        "--full-face",
        action="store_true",
        help="Draw full face mesh instead of sparse features"
    )
    
    parser.add_argument(
        "--model-complexity",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="Model complexity (0=fast, 1=balanced, 2=accurate)"
    )
    
    return parser.parse_args()


def main():
    """Main pipeline loop."""
    args = parse_args()
    
    # Configuration based on args
    enable_pose = not args.no_pose
    enable_hands = not args.no_hands
    enable_face = not args.no_face
    
    print("\n" + "=" * 50)
    print("  NeuroGym Vision Pipeline")
    print("=" * 50)
    print(f"  Camera: {args.camera}")
    print(f"  Resolution: {args.width}x{args.height}")
    print(f"  Pose Tracking: {'ON' if enable_pose else 'OFF'}")
    print(f"  Hand Tracking: {'ON' if enable_hands else 'OFF'}")
    print(f"  Face Tracking: {'ON' if enable_face else 'OFF'}")
    print(f"  Model Complexity: {args.model_complexity}")
    print("=" * 50)
    print("  Press 'Q' to quit")
    print("=" * 50 + "\n")
    
    # Initialize components
    with VideoProcessor(
        source=args.camera,
        width=args.width,
        height=args.height
    ) as video_processor:
        
        with HolisticDetector(
            enable_pose=enable_pose,
            enable_hands=enable_hands,
            enable_face=enable_face,
            model_complexity=args.model_complexity
        ) as detector:
            
            drawer = CyberpunkDrawer(
                draw_pose=enable_pose,
                draw_hands=enable_hands,
                draw_face=enable_face,
                glow_effect=not args.no_glow,
                sparse_face=not args.full_face
            )
            
            # FPS tracking
            fps = 0.0
            fps_update_interval = 10
            frame_times = []
            frame_count = 0
            
            # Get actual frame size
            frame_width, frame_height = video_processor.get_frame_size()
            
            print(f"Actual resolution: {frame_width}x{frame_height}")
            print("Starting capture...\n")
            
            while True:
                frame_start = time.perf_counter()
                
                # Read frame in RGB for MediaPipe
                frame_rgb = video_processor.read_frame()
                
                if frame_rgb is None:
                    print("Failed to read frame, exiting...")
                    break
                
                # Process frame with MediaPipe
                results = detector.process(frame_rgb)
                
                # Extract structured landmarks (for future use)
                landmarks = detector.extract_landmarks(results)
                
                # Convert to BGR for display and drawing
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                
                # Draw cyberpunk overlays
                frame_display = drawer.draw(frame_bgr, results)
                
                # Calculate FPS
                frame_time = time.perf_counter() - frame_start
                frame_times.append(frame_time)
                frame_count += 1
                
                if frame_count % fps_update_interval == 0:
                    avg_time = sum(frame_times[-fps_update_interval:]) / fps_update_interval
                    fps = 1.0 / avg_time if avg_time > 0 else 0.0
                
                # Draw FPS
                CyberpunkDrawer.draw_fps(frame_display, fps)
                
                # Draw module status
                status_parts = []
                if enable_pose:
                    status_parts.append("POSE")
                if enable_hands:
                    status_parts.append("HANDS")
                if enable_face:
                    status_parts.append("FACE")
                status = " | ".join(status_parts) if status_parts else "NO MODULES"
                CyberpunkDrawer.draw_status(frame_display, f"Active: {status}")
                
                # Display frame
                cv2.imshow("NeuroGym - AI Fitness Coach", frame_display)
                
                # Check for quit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("\nExiting...")
                    break
    
    # Cleanup
    cv2.destroyAllWindows()
    
    # Print summary
    if frame_times:
        avg_fps = len(frame_times) / sum(frame_times)
        print(f"\nSession Summary:")
        print(f"  Total Frames: {frame_count}")
        print(f"  Average FPS: {avg_fps:.1f}")
    
    print("NeuroGym pipeline closed successfully.")


if __name__ == "__main__":
    main()
